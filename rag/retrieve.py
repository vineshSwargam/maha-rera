# retrieve.py: the dot-product or cosine similarity search.
from rag.embed import embed_query
from rag.config import DEVICE
from sentence_transformers import SentenceTransformer
import torch
from time import perf_counter as timer
from rag.utils import print_wrapped

def retrieve_relevant_resources(
    query: str,
    embeddings: torch.tensor,
    dot_score_util,
    model: SentenceTransformer,
    n_resources_to_return: int=5,
    print_time: bool=True
):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = dot_score_util(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on ({len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return);

    return scores, indices



def print_top_results_and_scores(
    query: str,
    embeddings: torch.tensor,
    dot_score_util,
    model: SentenceTransformer,
    pages_and_chunks: list[dict],
    n_resources_to_return: int=5,
    print_time: bool = False
):
    """
    Finds relevant passages given a query and prints them out along with their scores.
    """
    scores, indices = retrieve_relevant_resources(
        query = query,
        embeddings = embeddings,
        dot_score_util = dot_score_util,
        model = model,
        n_resources_to_return = n_resources_to_return,
        print_time = print_time
    )

    # Loop through zipped together scores and indices from torch.topk
    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")

