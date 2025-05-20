import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
import os
from typing import Union, List

load_dotenv()

class LLMHelper:
    def __init__(self, model_id:str= "RedHatAI/Meta-Llama-3-8B-Instruct-quantized.w4a16", model_type:str="hf", token=None, bnb_config:BitsAndBytesConfig=None):
        self.model_id= model_id
        self.model_type= model_type
        self.config= bnb_config

        if model_type == "hf":
            self.tokenizer= AutoTokenizer.from_pretrained(model_id, token=token, device_map="auto")
            self.model= AutoModelForCausalLM.from_pretrained(model_id, token=token, device_map="auto") #quantization_config=self.bnb_config)
        
        elif model_type == "pipeline":
            self.model= pipeline("text-generation",
                                 model=model_id,
                                 model_kwargs={"torch_dtype":torch.bfloat16},
                                 quantization_config=bnb_config,
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
        if "mistral" in self.model_id.lower():
            full_prompt= [
                {"role":"system", "content": "you are a helpful assistant that answers the user's question considering the following context:\n %s" % context}
            ]
            return full_prompt
        if "llama" in self.model_id.lower():

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
        assert isinstance(prompt, (str, list)), f"Prompt is not viable. {prompt}"
        # if "mistral" in self.model_id.lower():
        #     output= self.model(prompt)
        #     return output
        # elif "llama" in self.model_id.lower():
        inputs= self.tokenizer(prompt, return_tensors='pt')
        inputs= {k:v.to(self.model.device) for k, v in inputs.items()}
        print(f"[DEBUG]: Inputs are on device {inputs['input_ids'].device}")
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        print(f"[DEBUG]: Outputs generated on {output_ids.device}")

        # Slice out the generated part
        generated_tokens = output_ids[0][inputs['input_ids'].shape[-1]:]

        # Decode only the new tokens
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return answer
        # elif self.model_type == "other":
        #     pass

#login(token=os.environ.get("HF_TOKEN"))

bnb_config= BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

