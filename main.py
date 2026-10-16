# main.py: wires the steps together:
from dotenv import load_dotenv

load_dotenv()   # reads .env into the environment

from rag.ingest import download_document, open_and_read_pdf, pdf_path, url, process_sentences, split_pages_into_chunks, filter_short_chunks, load_folder
from rag.embed import embedding_model, sentence_transformer_util, test_embedding_model, embed_chunks, save_embeddings, view_embeddings, retrieve_embeddings_for_search
from rag.retrieve import retrieve_relevant_resources, print_top_results_and_scores
from rag.llm_resources import init_model
from rag.generate import ask
from pprint import pprint
import pandas as pd
import random
import os, shutil

NUTRITION_EMBEDDINGS_PATH = "embeddings.csv"
RERA_EMBEDDINGS_PATH = "rera_embeddings.csv"

EMBEDDINGS_PATH = RERA_EMBEDDINGS_PATH

def verify_pre_requisites():
    tessdata = os.getenv("TESSDATA_PREFIX")
    if not shutil.which("tesseract") or not tessdata or not os.path.exists(os.path.join(tessdata, "eng.traineddata")):
        raise RuntimeError("Tesseract / eng.traineddata not found; OCR unavailable")

def print_embeddings(chunk_pages):
    chunk_pages_df = pd.DataFrame(chunk_pages)
    pprint(chunk_pages_df.head())
    pprint(chunk_pages_df.describe())

    # embedded_chunks = embed_chunks(chunk_pages, "sentence_chunk", "embedding");
    # embedded_chunks_df = pd.DataFrame(embedded_chunks)
    # pprint(embedded_chunks_df.head())
    # pprint(embedded_chunks_df.describe())

def _build_embeddings():
    download_document(url, pdf_path)
    page_number_offset = 41
    pages = open_and_read_pdf(pdf_path, page_number_offset)

    print(f"Read {len(pages)} pages")
    # print(pages[50])  # look at one page to sanity-check
    pages = process_sentences(pages);
    # pprint(pages[50]);


    # More than 25% of pages are above 384 tokens limitation of our embedding model
    # sentence-transformers all-mpnet-base-v2, see: https://www.sbert.net/docs/pretrained_models.html
    # lets split it
    chunk_size = 10

    chunk_pages = split_pages_into_chunks(pages, chunk_size)
    chunk_pages = filter_short_chunks(chunk_pages, "chunk_token_count", 30)
    #  print_embeddings(chunk_pages)

    embedded_chunks = save_embeddings(pd, chunk_pages, "sentence_chunk", "embedding", EMBEDDINGS_PATH)
    # view_embeddings(pd, EMBEDDINGS_PATH)


def build_embeddings():
    load_folder('./birlaNiyara')
    # pages = load_folder('./birlaNiyara')

    # print(f"Read {len(pages)} pages")
    # print(pages[50])  # look at one page to sanity-check
    # pages = process_sentences(pages);
    # pprint(pages[50]);


    # More than 25% of pages are above 384 tokens limitation of our embedding model
    # sentence-transformers all-mpnet-base-v2, see: https://www.sbert.net/docs/pretrained_models.html
    # lets split it
    # chunk_size = 10

    # chunk_pages = split_pages_into_chunks(pages, chunk_size)
    # chunk_pages = filter_short_chunks(chunk_pages, "chunk_token_count", 30)
    # #  print_embeddings(chunk_pages)

    # embedded_chunks = save_embeddings(pd, chunk_pages, "sentence_chunk", "embedding", EMBEDDINGS_PATH)
    # view_embeddings(pd, EMBEDDINGS_PATH)

queries = [
    'good foods for protein',
    'What are the macro nutrients and what roles do they play in the human body?',
    'name five fiber containing foods',
    'What role does fiber play in digestion? Name 5 fiber-containing foods ',
    'Describe the process of digestion and absorption of nutrients in the human body?'
]


def main():
    verify_pre_requisites()

    if not os.path.exists(EMBEDDINGS_PATH):
        print('No saved embeddings, building them..')
        build_embeddings()

    retrieved_embeddings = retrieve_embeddings_for_search(pd, EMBEDDINGS_PATH, 'embedding')
    embeddings = retrieved_embeddings['embeddings']
    chunk_pages = retrieved_embeddings['chunk_pages']

    llm = init_model(
        use_quantization_config = False,
    )

    def retrieve_and_answer():
        count_of_resources_to_retrieve = 5


        scores, indices = retrieve_relevant_resources(
            query=query,
            embeddings=embeddings,
            dot_score_util=sentence_transformer_util.dot_score,
            model=embedding_model,
            n_resources_to_return=count_of_resources_to_retrieve,
            print_time=False,
        )

        answer = ask(
            query=query,
            pages_and_chunks=chunk_pages,
            llm_model=llm["llm_model"],
            tokenizer=llm["tokenizer"],
            scores=scores,
            indices=indices,
            temperature=0.9,
            max_new_tokens=500
        )
        
        # print(f"\n{answer}")


    while True:
        query = input("\nAsk me a nutrition question (or 'q' to quit): ").strip()

        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:          # user just pressed Enter
            continue

        retrieve_and_answer()

if __name__ == "__main__":
    main()
    