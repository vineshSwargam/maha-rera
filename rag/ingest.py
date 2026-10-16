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
import re


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


def looks_like_drawing(text: str) -> bool:
    words = text.split()
    if len(words) < 30:
        return False
    short_caps = sum(1 for w in words if len(w) <= 4 and w.isupper())
    unique_ratio = len(set(words)) / len(words)
    return short_caps / len(words) > 0.4 or unique_ratio < 0.3


def ocr_page(page) -> str:
    # Run OCR on a scanned page (image only, no text layer)
    try:
        textpage = page.get_textpage_ocr(language="eng", dpi=300, full=True)
        return page.get_text(textpage=textpage)
    except RuntimeError as e:
        # Tesseract missing, or its language data couldn't be found
        print(f"[OCR failed on page {page.number}]: {e}")
        return ""



def summarize_drawing(text: str, source: str) -> str:
    # Turn floor-plan label soup into a short, searchable description

    words = text.split()

    # Title-block style phrases: runs of 3+ words that aren't all short abbreviations
    phrases = re.findall(r"(?:[A-Za-z&().,'-]{3,}\s+){2,}[A-Za-z&().,'-]{3,}", text)
    title_block = " · ".join(dict.fromkeys(p.strip() for p in phrases))  # dedupe, keep order

    # Most common labels (LOBBY, CAR LIFT, RAMP...), ignoring 1–2 letter codes and numbers
    labels = [w for w in words if len(w) >= 3 and w.isalpha()]
    top_labels = ", ".join(w for w, _ in Counter(labels).most_common(15))

    return (
        f"Architectural drawing from {source}. "
        f"Title block: {title_block[:1000]}. "
        f"Labels on drawing: {top_labels}."
    )


def get_page_text(page, source: str, page_number: int):
    text = page.get_text()
    text = fix_private_use_chars(text)

    print('                                                                                         ')
    print('                                                                                         ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if looks_like_drawing(text):
        print(f'DRAWING PAGE - {source} - Pg {page_number}')
        text = summarize_drawing(text, source=source)   # title block + key labels
        print('DRAWING SUMMARISED')
        print(text)
    elif len(text.strip()) < 50:
        print(f'OCR PAGE - {source} - Pg {page_number}')
        text = ocr_page(page) # scanned pages
        print(text)
    
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
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = get_page_text(page, pdf_path, page_number)
        text = text_formatter(text);

        pages_and_texts.append({
            "page_number": page_number - page_number_offset,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(".")),
            "page_token_count": len(text)/4, # assuming 4 chars is one token
            "text": text
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
    for pdf_path in tqdm(sorted(Path(folder).glob("*.pdf"))[:25]):
        print('                                                                                         ')
        print('                                                                                         ')
        print('                                                                                         ')
        print('====================================================================================')
        print('======= N E W    F I L E ===========================================================')
        print('====================================================================================')
        print(f'Source: {pdf_path.name}')
        pages = open_and_read_pdf(str(pdf_path), page_number_offset=0)
        for page in tqdm(pages):
            page["source"] = pdf_path.name     # keep the filename with every page
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


