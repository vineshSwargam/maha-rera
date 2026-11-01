# Build Journey: Local RAG over real-estate documents

The problems we hit while building this, why they happened, and how we fixed them. Ordered roughly by when we ran into them.

Each entry has a **Problem**, its **Cause**, the **Fix**, **Where** the fix lives (file → function, line range), and a **Takeaway** where there's a lesson worth keeping.

> Line ranges were correct as of **2026-11-01** and will drift as the code changes. The file and function names are the stable reference.

---

## Phase 1: Environment setup (macOS, M-series)

### 1. `python` command not found
- **Problem:** `python` failed in the terminal.
- **Cause:** macOS only ships `python3` (Apple's 3.9.6), which is too old for this project's dependencies.
- **Fix:** Installed Python 3.11 with Homebrew and created a **per-project venv** at the project root.
- **Where:** `venv/` (project root), `requirements.txt`. No code change.
- **Takeaway:** Every project gets its own venv (Python's equivalent of `node_modules`), so dependencies never conflict across projects.

### 2. Editor showed "import could not be resolved"
- **Problem:** Every import was underlined, and the editor warned about the interpreter.
- **Cause:** The venv had been created one folder up (`projects/venv`), so the editor fell back to the system Python 3.9.
- **Fix:** Recreated the venv inside the project root and selected `./venv/bin/python` as the interpreter.
- **Where:** `venv/` (project root) and the editor's interpreter setting.

### 3. No hover, autocomplete or go-to-definition
- **Problem:** Cmd+click and hover did nothing.
- **Cause:** The editor is **Cursor**, a VS Code fork, where Microsoft's Pylance isn't available. Three language servers were installed at once (Cursor Pyright, basedpyright, MS Python) and conflicted, and uninstalled extensions stayed loaded until a full restart.
- **Fix:** Kept only **basedpyright**, added a config pointing it at the venv, and fully restarted the editor.
- **Where:** `pyrightconfig.json` (project root): `venvPath`, `venv`, `typeCheckingMode`.
- **Takeaway:** Use one language server, and point it at the venv explicitly.

### 4. Type checker false positive on PyMuPDF
- **Problem:** `enumerate(doc)` was flagged as "not iterable".
- **Cause:** `fitz.Document` is iterable through the old `__getitem__` protocol, which type checkers don't recognise.
- **Fix:** Relaxed the type-checking mode. Iterating by index (`doc[i]`) is the type-safe alternative.
- **Where:** `pyrightconfig.json` → `typeCheckingMode`. The loop itself is in `rag/ingest.py` → `open_and_read_pdf()` (L200–246).

---

## Phase 2: From notebook to a modular Python project

### 5. Structuring the code
- **Problem:** All the logic lived in one Jupyter notebook.
- **Fix:** Split it into a package with one module per pipeline stage, with an entry point that wires them together.
- **Where:** `rag/config.py` (settings), `rag/ingest.py` (PDF → text → chunks), `rag/embed.py` (vectors), `rag/retrieve.py` (search), `rag/generate.py` (prompt + LLM), `rag/llm_resources.py` (model loading), `main.py` (entry point).
- **Takeaway:** Swapping the LLM, vector store or chunker now touches only one module.

### 6. Silent bugs from passing arguments by position
- **Problem:** Three separate bugs, all from positional arguments:
  - `check_filter_short_chunks(docs, 30, "key")` passed arguments in the wrong order, causing `KeyError: 30`.
  - `SentenceTransformer(name, DEVICE)` put `DEVICE` into the `modules` parameter, so the device setting was silently ignored.
  - `model.encode(text, batch_size, to_tensor)` put those values into `prompt_name` and `prompt`.
- **Fix:** **Only the first argument goes by position; everything else by keyword.**
- **Where:**
  - `main.py` → `build_embeddings()` (L36–59): the `filter_short_chunks(...)` call
  - `rag/embed.py`: module-level `SentenceTransformer(..., device=DEVICE)` (L11–14)
  - `rag/embed.py` → `embed_chunks_in_batches()` (L44–60)
- **Takeaway:** Python matches positional arguments by order, not by name, and the wrong order often doesn't raise an error.

### 7. Habits carried over from JS/TS
| Bug | Cause | Fix | Where |
|---|---|---|---|
| `return { a, b }` | In Python that's a **set**, not an object | Return a tuple, dict or dataclass | `rag/llm_resources.py` → `check_processing_power()` (L28–52), `check_model_compatibility()` (L53–80) |
| `a, b = func()` when `func` returns a dict | Unpacking a dict gives its **keys** (silent bug) | Index the dict | `rag/llm_resources.py` → `init_model()` (L147–188) |
| `f"{item["key"]}"` | Python 3.11 doesn't allow the same quote type inside an f-string | Use `item['key']` | `rag/ingest.py` → `split_pages_into_chunks()` (L304–334) |
| `else if`, `===`, `false` | JS syntax | `elif`, `==`, `False` | `rag/llm_resources.py` → `check_processing_power()`; `main.py` → `main()` |
| `name: value` in a call | `:` is for type hints | `name=value` | `main.py` → `main()` (L78–136): the `retrieve_relevant_resources(...)` and `init_model(...)` calls |
| A function containing only a comment | Comments don't count as a body | Add `pass` | `rag/ingest.py` (an early `process_document` stub) |
| `param: str = None` | The default contradicts the type | `str \| None = None` | `rag/llm_resources.py` → `init_model()` (L147–188) |

### 8. Default values that broke import
- **Problem:** `def f(device=Device)` raised a `NameError` when the module was imported.
- **Cause:** Default values are evaluated when the function is **defined**, not when it's called, and the name was misspelled (`Device` instead of `DEVICE`).
- **Fix:** Corrected the name. Names are case-sensitive.
- **Where:** `rag/llm_resources.py` → `check_processing_power()` (L28–52), `load_llm()` (L81–123), `init_model()` (L147–188).

---

## Phase 3: Core RAG pipeline (nutrition textbook)

### 9. Pages too long for the embedding model
- **Problem:** More than 25% of pages exceeded the 384-token limit of `all-mpnet-base-v2`.
- **Fix:** Split pages into sentences with spaCy, group them 10 sentences per chunk, and filter out chunks under 30 tokens.
- **Where:** `rag/ingest.py` → `process_sentences()` (L283–299), `split_sentence_list_into_chunks()` (L300–303), `split_pages_into_chunks()` (L304–334), `filter_short_chunks()` (L342–372); called from `main.py` → `build_embeddings()` (L36–59).
- **Takeaway:** Always measure chunk sizes against the embedding model's limit.

### 10. Re-embedding on every run
- **Problem:** Each run re-chunked and re-embedded the whole book.
- **Fix:** Save **chunks and vectors together** in one CSV. If the file exists, load it and skip the build.
- **Where:** `rag/embed.py` → `save_embeddings()` (L61–69), `retrieve_embeddings_for_search()` (L75–88); the existence check is in `main.py` → `main()` (L78–136).
- **Takeaway:** The vector, the text and the metadata belong in the same record. That's the same model a vector DB uses, and it guarantees search indices map back to the right chunks.

### 11. Embedding twice / passing the wrong type
- **Problem:** Chunks were embedded once in `embed_chunks` and again inside `save_embeddings`. Separately, a `{embeddings, chunk_pages}` dict was passed where a tensor was expected (`Could not infer dtype of dict`).
- **Fix:** Embed only inside `save_embeddings` (the `embed_chunks` call is commented out), and pull the tensor and chunk list out of the dict explicitly.
- **Where:** `rag/embed.py` → `save_embeddings()` (L61–69); `main.py` → `build_embeddings()` (L36–59), `main()` (L78–136).

---

## Phase 4: Running LLMs locally on a Mac

### 12. Code written for NVIDIA GPUs
- **Problem:** The reference notebook uses `bitsandbytes` and `flash-attn`, which are CUDA-only.
- **Fix:**
  - Detect the device as `cuda` → `mps` → `cpu`.
  - Import CUDA-only libraries **inside** the CUDA branch.
  - Use the `sdpa` attention implementation elsewhere.
  - Add platform markers in `requirements.txt`.
- **Where:** `rag/config.py` → `DEVICE` (L9); `rag/llm_resources.py` → `load_llm()` (L81–123); `requirements.txt` (L15–16, `; sys_platform == "linux"`).
- **Takeaway:** One code path adapts to the hardware; test the CUDA branch on Colab or Kaggle.

### 13. Gated Hugging Face models and secrets
- **Problem:** Gemma requires accepting a license and authenticating.
- **Fix:** Use `HF_TOKEN` from `.env` via `python-dotenv` locally, and a platform secret in deployment. The license acceptance belongs to the account, not the machine.
- **Where:** `.env`; `main.py` (L2–4, `load_dotenv()`); `.gitignore` (L2, `.env`).
- **Incident:** `.env` had been **staged** in git before it was ignored. Unstaged it with `git rm --cached` before any commit.

### 14. Model "downloading again"
- **Problem:** The terminal run started a fresh multi-GB download.
- **Cause:** The notebook had auto-selected **Gemma 7B**; `main.py` hardcoded **Gemma 2B**, a different model.
- **Where:** `main.py` → `main()`: the `init_model(...)` call; `rag/llm_resources.py` → `init_model()` (L147–188), `check_model_compatibility()` (L53–80).
- **Takeaway:** The HF cache (`~/.cache/huggingface`) is shared across the notebook, terminal and venvs. Downloads are per model, not per tool.

### 15. Gemma produced gibberish ("The / Please / TheA…")
- **Cause:**
  1. **float16 overflow.** Gemma's activations overflow in fp16.
  2. **The prompt was too long.** Oversized chunks pushed it toward Gemma's 8k context window, which also made generation slow.
- **Fix:** Switch to `bfloat16`, retrieve fewer and smaller chunks, and trim the few-shot examples.
- **Where:** `rag/llm_resources.py` → `load_llm()` (L81–123): `torch_dtype` (L88), `bnb_4bit_compute_dtype` (L94); `main.py` → `main()`: `count_of_resources_to_retrieve`; `rag/generate.py` → `prompt_formatter()` (L6–49).
- **Status:** ⚠️ **Not applied yet.** L88 and L94 still use `torch.float16`.

---

## Phase 5: Collecting the MahaRERA documents

### 16. Scraping a JavaScript-driven government site
- **Problem:** The PDF "links" were JavaScript buttons that call an authenticated API. A fresh tab triggered a CAPTCHA.
- **Fix:** Worked inside the user's verified browser tab, hooked into the page's own requests to capture each `documentId`, and let the page fetch every file with its own session.
- **Where:** One-off browser scripting, not part of the project code. Output: `birlaNiyara/`.

### 17. Rate limiting and browser lock-up
- **Problem:** Burst requests got `403: Restricted due to multiple requests (DDoS)`. Triggering browser downloads opened dozens of Save dialogs and froze Chrome.
- **Fix:** **Fetch one file at a time** with a pause between each, and send the files to a **temporary local receiver on `127.0.0.1`** instead of Chrome's downloads. That meant no dialogs and no interference with other work.
- **Where:** One-off (a temporary receiver script, since deleted). Output: `birlaNiyara/`.
- **Result:** 115 files, de-duplicated by content hash to **98 unique PDFs**.
- **Takeaway:** Be polite to public servers, and keep automation out of the user's way.

---

## Phase 6: Making messy real-estate PDFs searchable

### 18. Retrieval was poor on the RERA documents
- **Diagnosis:**
  - **52 of 98 files (284 of 462 pages) were scanned images**, with no text layer, so they were never embedded.
  - Sentence splitting fails on forms and tables (no full stops), so chunks reached 2,500+ tokens and were silently cut off at 384.
  - CAD drawings produced label soup.
  - Chunks had no source filename.
- **Fix (in progress):** OCR (entries 19–20), `source` and `method` metadata on every page and chunk, and token-based chunking (open).
- **Where:**
  - `rag/ingest.py` → `open_and_read_pdf()` (L200–246): adds `source` and `method`
  - `rag/ingest.py` → `split_pages_into_chunks()` (L304–334): copies them onto each chunk
  - `rag/ingest.py` → `load_folder()` (L247–282)
  - `main.py` → `build_birla_embeddings()` (L60–77)

### 19. Garbled text in the PDF's own text layer
- **Problem:** Some pages extracted as `…` or `\x06\x12\x11…`.
- **Cause:**
  1. Symbol fonts that store ASCII shifted by `0xF000`, which can be decoded.
  2. Fonts with **no Unicode mapping** (raw glyph IDs, no consistent offset), which can't be decoded.
- **Fix:** Stop guessing "drawing vs scan". Ask one question: **can the text layer be trusted?** Measure the share of clean characters, and OCR anything under 98% clean or nearly empty.
- **Where:** `rag/ingest.py` → `fix_private_use_chars()` (L57–62), `text_quality()` (L63–77), `needs_ocr()` (L78–81), `get_page_text()` (L176–199).
- **Result:** 292 empty pages and about 63 garbled pages are routed to OCR; about 107 pages keep their text layer.

### 20. OCR output on drawings is noisy
- **Problem:** Tesseract reads line art and hatching as characters (`|`, `TTT`, `SSS`).
- **Fix:** Keep only lines with real words (letter ratio, at least one 3-letter word, de-duplicated).
- **Where:** `rag/ingest.py` → `ocr_page()` (L82–91), `clean_ocr_text()` (L92–111); applied in `get_page_text()` (L176–199).
- **Next:** An LLM summary per page, and a vision model for floor plans.

### 21. "How many flats are sold?" can't be answered by RAG
- **Cause:** It's a **counting** question across 5–6 pages of table rows. Retrieval only hands the LLM the top few chunks, and OCR flattens table structure.
- **Fix:** **Extract tables into structured data** and answer numeric questions with code. Use RAG for documents and pandas/SQL for numbers.
- **Where:** `rag/ingest.py` → `open_and_read_pdf()` (L200–246) writes each detected table to `tables/<file>_p<page>.csv`.

### 22. img2table accuracy was about 1 in 10
- **Problem:** Clearly visible cells came back as `None`. On a test page: 0/21 flat numbers and 2–7/21 statuses.
- **Cause:** It detected the grid but OCR'd the whole page, then mis-assigned words to cells.
- **Fix:** Built our own reader: **OpenCV morphology to find the grid lines**, erase them, crop each cell, and OCR it as a single line (`tesseract --psm 7`).
- **Where:** `rag/ingest.py` → `_line_positions()` (L112–127), `_ocr_cell()` (L128–141), `extract_tables()` (L142–175).
- **Result:** **20/21 flat numbers and 21/21 statuses**. A page with no grid returns `None`, so the same function also tells table pages from the rest.

### 23. Crash: `cv2.imwrite` on an empty image
- **Cause:** Double or thick borders were detected as two lines a few pixels apart, so cropping the padding left a zero-pixel cell.
- **Fix:** Merge lines less than 20px apart, and skip cells smaller than 8px.
- **Where:** `rag/ingest.py` → `_line_positions()` (gap check at L120); `_ocr_cell()` (size guard at L129).

### 24. Cell-by-cell OCR is slow
- **Cause:** One Tesseract process per cell (about 130 cells per page).
- **Fix:** Skip blank cells (less than 0.5% ink) and reject grids over 400 cells (those are drawings, not tables).
- **Where:** `rag/ingest.py` → `_ocr_cell()` (blank check at L131); `extract_tables()` (cell cap at L160).
- **Status:** Parallel OCR with a thread pool in `extract_tables()` is **not applied yet**.

### 25. Partly filled and faint table pages lost their columns
- **Problem:** The last pages of the sold reports (for example 4 rows) gave **0 columns**, so no CSV was written (`202403_Sold_p6`, `CTIL_N2_Sep_24_Sold (2)_p6`). Faint scans merged columns: FY25 Q4 p3 put status and date in one cell (`Sold 10-Oct-24`), and p5 found only about 3 columns.
- **Cause:** A vertical line had to cover 20% of the **page** height, but a short table's column lines are shorter than that.
- **Fix:** Measure the column lines relative to the **table's own height** (between the first and last horizontal line, at 25%), with a shorter kernel for broken lines.
- **Where:** `rag/ingest.py` → `extract_tables()` (L142–175): vertical kernel `(1, 40)` (L150), table-relative column detection (L156).
- **Result:** All pages of all 5 sold-inventory PDFs now detect 6 columns.

---

## Open items / next steps
- [ ] Apply the `bfloat16` fix in `rag/llm_resources.py` → `load_llm()` (entry 15).
- [ ] Parallel cell OCR in `rag/ingest.py` → `extract_tables()` (entry 24).
- [ ] Token-based chunking (about 300 tokens with overlap) to replace `process_sentences` / `split_pages_into_chunks`.
- [ ] LLM summary per OCR'd page (for drawings, forms and tables).
- [ ] A router that sends counting questions to the extracted tables and everything else to RAG.
- [ ] Add the MahaRERA project summary page (unit counts, dates) as a document.
- [ ] Try a newer small model through Ollama (faster and better than Gemma 7B on a Mac).
- [ ] Add `embeddings.csv`, `rera_embeddings.csv`, `tables/` and `__pycache__/` to `.gitignore`.
- [ ] Download the registration certificate manually (its button didn't trigger a download).
- [ ] Revoke Chrome's local-network permission for maharerait.maharashtra.gov.in.
