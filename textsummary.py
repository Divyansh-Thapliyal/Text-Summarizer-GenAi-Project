import torch
import gradio as gr

# This code was in huggingface itself. (Next 2 lines.)
# Use a pipeline as a high-level helper
from transformers import pipeline

text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# text="The quick brown fox jumps over the lazy dog. " * 10
# print(text_summary(text))

#till here it is only console based. Lets use gradio to give it a GUI (graphic user interface) and make it a web app....

#Lets make a function bcz gradio needs a function to call.

def summary(input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

demo = gr.Interface(fn=summary, inputs="text", outputs="text")
demo.launch()