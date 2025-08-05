import torch
import gradio as gr

# This code was in huggingface itself. (Next 2 lines.)
# Use a pipeline as a high-level helper
from transformers import pipeline

text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text="The quick brown fox jumps over the lazy dog. " * 10

print(text_summary(text))
