import sys
import os
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv(dotenv_path=".env.test")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.llm import LLMHelper
assert torch.cuda.is_available(), "Cuda is not available."


def test_generation():
    assert torch.cuda.is_available(), "Cuda is not available."
    bnb_config= BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    llm= LLMHelper(model_id="RedHatAI/Meta-Llama-3-8B-Instruct-quantized.w4a16", model_type="hf", token=os.getenv("HF_TOKEN"), bnb_config=bnb_config)
    assert torch.device(llm.model.device).type == "cuda", "Model is not on CUDA"
    llm_prompt= llm.create_prompt("Nitamonkey is corny.", "What adjective would you use to describe Nitamonkey?")
    assert isinstance(llm_prompt, (str, list))


    answer= llm.generate_answer(llm_prompt)
    assert answer, "Nothing generated. Check generation parameters."
    assert "corny" in answer.lower()
