
from llama_cpp import Llama

class LLMHelper:
    def __init__(self, model_path:str="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"):
        self.model= Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=1024)
        self.model_path= model_path

    def create_prompt(self, prompt:str, context:str) -> str:
        """
        Returns a string formatted for use in Mistral
        
        Args:
            prompt (str): The instructions for Mistral
            context (str): Context pulled from a Vector DB.
        Returns:
            str: String with Mistral formatting
        """
        full_prompt= \
        """
        <s>[INST] You are a helpful assistant. Use the context below to answer the query posed by the user.

        Context:
        %s

        Query:
        %s
        [/INST]
        """ % (prompt, context)

        return full_prompt
    

    def generate_answer(self, prompt:str) -> str:
        """
        Returns an LLM-generated answer to the question provided.
        
        Args:
            prompt (str): Mistral formatted prompt.

        Returns:
            str: The model's first choice answer as a string.
        """

        answer= self.model(prompt, max_tokens=1024, temperature=0.1)

        return answer["choices"][0]["text"]
    

import os
model_path = os.path.abspath("models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
llm = Llama(model_path=model_path, n_gpu_layers=50)
print("GPU layers in use:", llm.model_params.n_gpu_layers)

# TODO: Install C++ Build Tools via Visual Studio Installer and NVCC via nvidia