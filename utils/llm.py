import torch
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv
import os
from typing import Union, List

load_dotenv()

class LLMHelper:
    def __init__(self, model_id:str= "meta-llama/Llama-3.1-8B", model_type:str="pipeline", token=None):
        self.model_id= model_id
        self.model_type= model_type
        
        if model_type == "pipeline":
            self.model= pipeline("text-generation",
                                 model=model_id,
                                 token=os.getenv("HF_API_TOKEN"),
                                 model_kwargs={"torch_dtype":torch.bfloat16},
                                 device_map='auto')
        

    def create_prompt(self, context:str, query:str) -> str:
        """
        Returns a string formatted for use with a HuggingFace LLM.
        
        Args:
            prompt (str): The instructions for the LLM.
            context (str): Context pulled from a Vector DB.
        Returns:
            str: String with context and query formatting.
        """
        if self.model_type == "mistral":
            full_prompt= [
                {"role":"system", "content": "you are a helpful assistant that answers the user's question considering the following context:\n %s" % context}
            ]
            return full_prompt
        if self.model_type == "llama":

            full_prompt= \
            """
            You are a helpful assistant. Use the context below to answer the query posed by the user.

            Context:
            %s

            Query:
            %s
            """ % (context, query)

            return full_prompt
    

    def generate_answer(self, prompt:Union[List, str]) -> str:
        """
        Returns an LLM-generated answer to the question provided.
        
        Args:
            prompt (str): Prompt including any context you require.

        Returns:
            str: The model's first choice answer as a string.
        """
        if "mistral" in self.model_id:
            output= self.model(prompt)
            return output
        elif "llama" in self.model_id:
            # tokenize the input prompt
            input= self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output= self.model.generate(**input, max_new_tokens=2048)

            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        elif self.model_type == "other":
            pass


llm= LLMHelper(token=os.getenv("HF_API_TOKEN"))

llm.generate_answer("Tell me a fun fact about Large Language Models")