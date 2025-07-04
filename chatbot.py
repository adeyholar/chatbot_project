from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator
import os

# Set Hugging Face cache directory
os.environ["HF_HOME"] = r"D:\AI\Models\huggingface"

class Chatbot:
    def __init__(self, model_path):
        # Initialize Accelerator for GPU optimization
        self.accelerator = Accelerator()
        self.device = self.accelerator.device  # cuda:0 if GPU available

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check GPU setup.")

        # Quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load tokenizer and model from local path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            quantization_config=quantization_config
        ).to(self.device)

        # Prepare model with Accelerator
        self.model = self.accelerator.prepare(self.model)

        # Initialize conversation history
        self.history = []

    def generate_response(self, user_input, max_length=50):
        # Append user input to history
        self.history.append(f"User: {user_input}")

        # Prepare context (last 3 turns)
        context = " ".join(self.history[-3:])
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Generate response with explicit parameters
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            length_penalty=0.65,
            no_repeat_ngram_size=3,
            do_sample=True,
            temperature=0.7
        )

        # Decode response with explicit cleanup
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Append response to history
        self.history.append(f"Bot: {response}")
        return response

    def get_history(self):
        return self.history

def main():
    # Load model from local path
    model_path = r"D:\AI\Models\blenderbot_400M"
    print(f"Loading chatbot from {model_path}")
    chatbot = Chatbot(model_path)

    # Interactive loop
    print("Start chatting! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")

    # Print conversation history
    print("\nConversation History:")
    for turn in chatbot.get_history():
        print(turn)

if __name__ == "__main__":
    main()