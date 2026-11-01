# ingest.py: the PyMuPDF reading and sentence/chunk splitting code.

from tqdm.auto import tqdm
from spacy.lang.en import English
from pprint import pprint
from pathlib import Path
from collections import Counter
import os
import requests
import fitz
import re
import pandas as pd
import random

import unicodedata
import subprocess
import tempfile
import cv2
import numpy as np


pdf_path = "data/human-nutrition-text.pdf"
url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

def download_document(url: str, filename: str):
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at {pdf_path}. Starting download...")

        # Enter the URL of the PDF

        # The local filename to save the downloaded file
        filename = pdf_path
        
        # Send a GET request to the URL
        response = requests.get(url);

        if response.status_code == 200:
            # Open the file and save it
            with open(filename, "wb") as file:
                file.write(response.content)

            print(f"PDF file downloaded and saved to {filename}")
        else:
            print(f"Failed to download PDF file. Status code: {response.status_code}")

    else:
        print(f"PDF file found at {pdf_path}")

# Ingest
def text_formatter(text: str) -> str:
    """Performs minor formatting on document text"""
    cleaned_text = text.replace("\n", " ").strip();

    # Potential further formattings can be added here
    return cleaned_text

def fix_private_use_chars(text: str) -> str:
    return "".join(
        chr(ord(c) - 0xF000) if 0xF020 <= ord(c) <= 0xF0FF else c
        for c in text
    )

def text_quality(text: str) -> float:
    """Fraction of non-space characters that are real, readable text (0.0–1.0)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    def is_clean(c: str) -> bool:
        cat = unicodedata.category(c)
        if cat.startswith("C"):                      # control / private-use / unassigned
            return False
        if cat.startswith("L") and ord(c) > 0x24F:   # letters outside Latin (Greek/Coptic junk)
            return False
        return True
    return sum(is_clean(c) for c in chars) / len(chars)


def needs_ocr(text: str, min_chars: int = 50, min_quality: float = 0.98) -> bool:
    return len(text.strip()) < min_chars or text_quality(text) < min_quality


def ocr_page(page) -> str:
    # Run OCR on a scanned page (image only, no text layer)
    try:
        textpage = page.get_textpage_ocr(language="eng", dpi=300, full=True)
        return page.get_text(textpage=textpage)
    except RuntimeError as e:
        # Tesseract missing, or its language data couldn't be found
        print(f"[OCR failed on page {page.number}]: {e}")
        return ""

def clean_ocr_text(text: str, min_alpha: int = 3, min_alpha_ratio: float = 0.5) -> str:
    """Drop OCR noise lines (|, TTT, ===) and repeated labels; keep lines with real words."""
    kept, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        letters = sum(c.isalpha() for c in line)
        if letters < min_alpha:
            continue
        if letters / max(len(line.replace(" ", "")), 1) < min_alpha_ratio:
            continue
        if not re.search(r"[A-Za-z]{3,}", line):
            continue
        if line.lower() in seen:
            continue
        seen.add(line.lower())
        kept.append(line)
    return "\n".join(kept)



def _line_positions(mask, axis: int, min_frac: float) -> list[int]:
    """Centres of long horizontal (axis=1) or vertical (axis=0) lines."""
    proj = mask.sum(axis=axis) / 255
    idx = np.where(proj > mask.shape[axis] * min_frac)[0]
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for x in idx[1:]:
        if x - prev > 20:
            out.append((start + prev) // 2)
            start = x
        prev = x
    out.append((start + prev) // 2)
    return out


def _ocr_cell(img, psm: int) -> str:
    if img.size == 0 or img.shape[0] < 8 or img.shape[1] < 8:
        return ""
    if (img < 128).mean() < 0.005:      # under 0.5% dark pixels → empty cell
        return ""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        path = f.name
    text = subprocess.run(["tesseract", path, "-", "--psm", str(psm)],
                          capture_output=True, text=True).stdout
    os.remove(path)
    return " ".join(text.split()).strip("| ")


def extract_tables(page, dpi: int = 300) -> list[list[str]] | None:
    """Read a ruled table from a page. Returns rows of cell strings, or None if no grid."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)))

    rows = _line_positions(horiz, axis=1, min_frac=0.3)
    if len(rows) < 3:
        return None
    # Look for column lines only between the table's top and bottom edges
    cols = _line_positions(vert[rows[0]:rows[-1], :], axis=0, min_frac=0.25)
    if len(cols) < 3:
        return None

    if (len(rows) - 1) * (len(cols) - 1) > 400:   # drawings, not tables
        return None

    clean = cv2.bitwise_not(cv2.subtract(bw, cv2.bitwise_or(horiz, vert)))
    pad = 6
    table = []
    for i in range(len(rows) - 1):
        psm = 6 if i == 0 else 7
        table.append([
            _ocr_cell(clean[rows[i] + pad:rows[i + 1] - pad, cols[j] + pad:cols[j + 1] - pad], psm)
            for j in range(len(cols) - 1)
        ])
    return table



def get_page_text(page, source: str, page_number: int):
    text = page.get_text()
    text = fix_private_use_chars(text)

    print('                                                                                         ')
    print('                                                                                         ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if not needs_ocr(text):
        return {"text": text, "method": "text-layer", "table": None}

    print(f"OCR: {source} p{page_number + 1}")
    table = extract_tables(page)                 # None if the page has no grid
    text = clean_ocr_text(ocr_page(page))
    return {"text": text, "method": "ocr", "table": table}
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('                                                                                         ')
    print('                                                                                         ')
    return text


def open_and_read_pdf(pdf_path: str, page_number_offset: int) -> list[dict]:
    print('                                                                                         ')
    print('                                                                                         ')

    # print('==============================================================================')
    print('======== NEW PAGE ======================================================================')
    # print('==============================================================================')
    doc = fitz.open(pdf_path)
    source = os.path.basename(pdf_path)
    os.makedirs("tables", exist_ok=True)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        result = get_page_text(page, source, page_number)
        text = text_formatter(result["text"])

        if result["table"]:
            pd.DataFrame(result["table"]).to_csv(
                f"tables/{Path(pdf_path).stem}_p{page_number + 1}.csv", index=False, header=False
            )

        pages_and_texts.append({
            "source": source,
            "method": result["method"],
            "page_number": page_number - page_number_offset,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(".")),
            "page_token_count": len(text) / 4,
            "text": text,
        })
    

    size_of_sample = len(pages_and_texts);
    print(f'Size: {size_of_sample}')

    if size_of_sample >= 10:
        pprint(random.sample(pages_and_texts, k=2))
    else:
        pprint(random.sample(pages_and_texts, k=size_of_sample))

    print('==============================================================================')
    print('==============================================================================')
    print('                                                                                         ')
    print('                                                                                         ')

    return pages_and_texts

def load_folder(folder: str) -> list[dict]:
    all_pages = []

    # for pdf_path in tqdm(sorted(Path(folder).glob("*.pdf"))[15:25]):
    for pdf_path in tqdm((Path(folder).glob("*.pdf"))):
        print('paths -', str(pdf_path))

    for pdf_path in tqdm((Path(folder).glob("*.pdf"))):
        if not 'sold' in str(pdf_path).lower():
            continue

        print('                                                                                         ')
        print('                                                                                         ')
        print('                                                                                         ')
        print('====================================================================================')
        print('======= N E W    F I L E ===========================================================')
        print('====================================================================================')
        print(f'Source: {pdf_path.name}')
        pages = open_and_read_pdf(str(pdf_path), page_number_offset=0)
        all_pages.extend(pages)
        print('====================================================================================')
        print('====================================================================================')
        print('                                                                                         ')
        print('                                                                                         ')
        print('                                                                                         ')
    return all_pages


# Text Processing
# documents = [
#   {
#     sentences: "Some sentence example and another word in the same sentence",
#     ....
#   },
#   { ... },
# ]
def process_sentences(documents):

    nlp = English()

    # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer 
    nlp.add_pipe("sentencizer")

    for item in tqdm(documents):
        item["sentences"] = list(nlp(item["text"]).sents)

        # Make sure all sentences are strings (the default type is a spaCy datatype)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        
        item["page_sentence_count_spacy"] = len(item["sentences"])

    return documents

def split_sentence_list_into_chunks(sentence_list: list[str], slice_size: int) -> list[list[str]]:
    return [sentence_list[i: i + slice_size] for i in range(0, len(sentence_list), slice_size)]


def split_pages_into_chunks(documents, slice_size: int):
    chunk_documents = []
    for item in tqdm(documents):
        # pprint(item['page_number'])
        # pprint(item['sentences'])
        item["sentence_chunks"] = split_sentence_list_into_chunks(item['sentences'], slice_size)
        # pprint('======== Sentence Chunks =======')
        # pprint(item['sentence_chunks'])
        # pprint('======== Sentence Chunks Count =======')
        item["sentence_chunks_count"] = len(item['sentence_chunks'])
        # pprint(item["sentence_chunks_count"])
        # pprint('======== END =========')

        for sentence_chunk in item["sentence_chunks"]: 
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka join the list of sentences into one paragraph
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" => ". A" (will work for any captial letter)

            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get some stats on our chunks
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 chars

            chunk_documents.append(chunk_dict)
    return chunk_documents

def check_filter_short_chunks(document_list, key: str, min_token_length: int):
    df = pd.DataFrame(document_list)

    # Show random chunks with under min_token_length tokens in length
    for row in df[df[key] <= min_token_length].sample(5).iterrows():
        print(f'Chunk token count: {row[1][key]} | Text: {row[1]["sentence_chunk"]}')

def filter_short_chunks(document_list, key: str, min_token_length: int):
    df = pd.DataFrame(document_list)
    # Filter our DataFrame for rows with under 30 tokens
    pages_and_chunks_over_min_token_len = df[df[key] > min_token_length].to_dict(orient="records")
    return pages_and_chunks_over_min_token_len







    


    
    

# Evaluate token metrics 
# Evaluate chunk sizes based on embedding model limits
# Decide which size chunks makes sense
# Filter small chunks and evaluate whether it is noise

# Make each chunk it's own full item
# Embed the chunks
# Save the embeddings

# Convert embeddings back to searchable torch entites
# RAG Search and Answer


