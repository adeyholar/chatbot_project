import requests
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class ChatbotAPI:
    def __init__(self, api_key, api_url="https://api-inference.huggingface.co/models/google/flan-t5-base", local_model_path=None, max_retries=3):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.history = []
        self.local_mode = False
        self.max_retries = max_retries

        # Set up HTTP session with retries
        self.session = requests.Session()
        retries = Retry(total=max_retries, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        # Initialize local model if provided
        if local_model_path:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            if not torch.cuda.is_available():
                print("CUDA unavailable. API mode only.")
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        local_model_path,
                        quantization_config=quantization_config
                    ).to(self.device)
                    self.model = self.accelerator.prepare(self.model)
                    self.local_mode = True
                    print(f"Loaded local model from {local_model_path}")
                except Exception as e:
                    print(f"Failed to load local model: {e}. Using API mode.")

    def generate_response(self, user_input, max_length=50):
        self.history.append(f"User: {user_input}")
        context = " ".join(self.history[-3:])

        # Try API first
        if not self.local_mode or self.max_retries > 0:
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
                response = self.session.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list) and result:
                    response_text = result[0].get("generated_text", "").replace(context, "").strip()
                    if not response_text:
                        raise ValueError("Empty response from API")
                else:
                    raise ValueError(f"Unexpected API response format: {result}")
                self.history.append(f"Bot: {response_text}")
                return response_text
            except requests.exceptions.RequestException as e:
                print(f"API request failed: {e}. Falling back to local model if available.")

        # Fallback to local model
        if not self.local_mode:
            raise RuntimeError("No local model available for fallback.")
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
        self.history.append(f"Bot: {response}")
        return response

    def get_history(self):
        return self.history

def main():
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY not found in .env")

    # Initialize chatbot with local fallback
    local_model_path = r"D:\AI\Models\blenderbot_1B"
    print("Initializing Hugging Face API chatbot with local fallback")
    chatbot = ChatbotAPI(api_key, local_model_path=local_model_path)

    print("Start chatting! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        try:
            response = chatbot.generate_response(user_input)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")

    print("\nConversation History:")
    for turn in chatbot.get_history():
        print(turn)

if __name__ == "__main__":
    main()