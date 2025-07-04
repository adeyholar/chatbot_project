from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator
import os

# Set Hugging Face cache directory
os.environ["HF_HOME"] = r"D:\AI\Models\huggingface"

# Initialize Accelerator for GPU optimization
accelerator = Accelerator()

# Model selection
model_name = "facebook/blenderbot-400M-distill"
print(f"Loading model: {model_name}")

# Step 1: Update Quantization with BitsAndBytesConfig
# This replaces the deprecated load_in_4bit argument
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Recommended 4-bit quantization type
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for computation if your GPU supports it, else use torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with the new quantization configuration
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=quantization_config, # Pass the BitsAndBytesConfig object
    device_map="cuda:0"   # CHANGED: Explicitly map to cuda:0 instead of "auto"
)

# Prepare model with Accelerator
model = accelerator.prepare(model)

# Verify model is on GPU
print(f"Model device: {next(model.parameters()).device}")

# Test tokenizer with sample input
sample_input = "Hello, how are you?"
inputs = tokenizer(sample_input, return_tensors="pt").to(accelerator.device)
print(f"Tokenized input: {inputs}")

# Test model inference
outputs = model.generate(
    **inputs,
    max_length=60,         # Adjust as needed
    min_length=20,
    num_beams=10,
    length_penalty=0.65,
    no_repeat_ngram_size=3
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Sample response: {response}")

# Save model and tokenizer locally
local_path = r"D:\AI\Models\blenderbot_400M"
tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)
print(f"Model and tokenizer saved to {local_path}")

print("Model and tokenizer loaded successfully!")