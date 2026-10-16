# config.py: every hardcoded value ("mps", chunk_size=10, the model name, file paths) goes here as a constant. It's the single place to change settings.
import torch

# TO USE ANY KEY ANYWHERE, THIS IS HOW TO RETRIEVE IT
# import os
# HF_TOKEN = os.getenv("HF_TOKEN")   # None if not set


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"