import requests
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator

class ChatbotAPI:
    def __init__(self, api_key, api_url="https://api-inference.huggingface.co/models/google/flan-t5-base", local_model_path=None):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.history = []
        self.local_mode = False
        if local_model_path:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Using API only.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                local_model_path,
                quantization_config=quantization_config
            ).to(self.device)
            self.model = self.accelerator.prepare(self.model)
            self.local_mode = True
            print(f"Loaded local model from {local_model_path}")

    def generate_response(self, user_input, max_length=50):
        self.history.append(f"User: {user_input}")
        context = " ".join(self.history[-3:])
        if self.local_mode:
            inputs = self.tokenizer(
                context,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                length_penalty=0.65,
                no_repeat_ngram_size=3,
                do_sample=True,
                temperature=0.7
            )
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        else:
            payload = {
                "inputs": context,
                "parameters": {
                    "max_length": max_length,
                    "num_beams": 5,
                    "length_penalty": 0.65,
                    "no_repeat_ngram_size": 3,
                    "do_sample": True,
                    "temperature": 0.7
                }
            }
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list) and result:
                    response = result[0].get("generated_text", "").replace(context, "").strip()
                    if not response:
                        raise ValueError("Empty response from API")
                else:
                    raise ValueError(f"Unexpected API response format: {result}")
            except requests.exceptions.HTTPError as e:
                print(f"API request failed: {e}. Falling back to local model if available.")
                if not self.local_mode:
                    raise RuntimeError("No local model available for fallback.")
                return self.generate_response(user_input, max_length)
        self.history.append(f"Bot: {response}")
        return response

    def get_history(self):
        return self.history