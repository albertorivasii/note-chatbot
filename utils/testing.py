import torch
from transformers import pipeline, AutoTokenizer, Auto
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

try:
    login(os.getenv("HF_TOKEN"))
except Exception as e:
    raise ValueError(e)
pipe= pipeline("text-generation",
                                 model= "meta-llama/Llama-3.1-8B",
                                 token=os.getenv("HF_TOKEN"),
                                 model_kwargs={"torch_dtype":torch.bfloat16},
                                 device_map='auto'
				)

print(pipe("Tell me 1 interesting fact about LLMs"))