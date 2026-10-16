# embed.py: the SentenceTransformer code and saving embeddings to CSV or disk.
from sentence_transformers import util, SentenceTransformer
from rag.config import DEVICE, EMBEDDING_MODEL_NAME
from tqdm.auto import tqdm
from pprint import pprint
import torch
import numpy as np
from typing import Literal


EMBEDDING_MODEL_NAME_OR_PATH = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME_OR_PATH, device=DEVICE)
sentence_transformer_util = util

def test_embedding_model():
    # Create a list of sentences
    sentences = [
        "The Sentence Transformer library provides an easy way to create embeddings.",
        "Sentences can be embedded one by one or in a list.",
        "I like horses!"
    ]

    # Sentences are encoded/embedded by caling model.encode()
    embeddings = embedding_model.encode(sentences)
    embeddings_dict = dict(zip(sentences, embeddings))

    # See the embeddings
    for sentence, embedding in embeddings_dict.items():
        print(f"Sentence: {sentence}")
        print(f"Embedding: {embedding}")
        print("")

def embed(val: str):
    return embedding_model.encode(val)

def embed_query(query: str, conv_to_tensor: bool = True, use: Literal["cpu", "mps", "cuda"] = "cpu"):
    return embedding_model.encode(query, convert_to_tensor=conv_to_tensor).to(use)

def embed_chunks(chunks_doc_list, embedding_key: str, embedding_store_key):
    for item in tqdm(chunks_doc_list):
        item[embedding_store_key] = embed(item[embedding_key])
    return chunks_doc_list;

def embed_chunks_in_batches(
    chunks_doc_list,
    chunk_key: str,
    batch_size: int,
    conv_to_tensor: bool = True
):
    text_chunks = [item[chunk_key] for item in chunks_doc_list]

    text_chunk_embeddings = embedding_model.encode(
        text_chunks,
        batch_size = batch_size, # you can experiment to find which batch size leads to best results
        convert_to_tensor = conv_to_tensor
    )
    return text_chunk_embeddings


# Save embeddings to file
def save_embeddings(pd, chunks_doc_list, embedding_key: str, embedding_store_key, save_path):
    for item in tqdm(chunks_doc_list):
        item[embedding_store_key] = embed(item[embedding_key])
        
    embeddings_df = pd.DataFrame(chunks_doc_list)
    embeddings_df.to_csv(save_path, index=False)
    return chunks_doc_list

# Import saved file and view 
def view_embeddings(pd, embeddings_file_path: str):
    embeddings = pd.read_csv(embeddings_file_path)
    pprint(embeddings.head())

    # Import texts and embedding df
def retrieve_embeddings_for_search(pd, embeddings_file_path, embedding_key: str):
    text_chunks_and_embedding_df = pd.read_csv(embeddings_file_path)

    # Convert embedding column back to np.array (it got converted to string when it saved to CSV)
    text_chunks_and_embedding_df[embedding_key] = text_chunks_and_embedding_df[embedding_key].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # Convert our embeddings into a torch.tensor
    embeddings = torch.tensor(np.stack(text_chunks_and_embedding_df[embedding_key].tolist(), axis=0), dtype=torch.float32).to(DEVICE)

    # Convert texts and embedding df to list of dicts
    chunk_pages = text_chunks_and_embedding_df.to_dict(orient="records")

    return { 'embeddings': embeddings, 'chunk_pages': chunk_pages }

