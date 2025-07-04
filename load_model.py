from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator
import os

# Set Hugging Face cache directory
os.environ["HF_HOME"] = r"D:\AI\Models\huggingface"

# Initialize Accelerator for GPU optimization
accelerator = Accelerator()
device = accelerator.device  # cuda:0 if GPU available

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check GPU setup.")

# Model selection
model_name = "facebook/blenderbot-400M-distill"
print(f"Loading model: {model_name}")

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with quantization
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=quantization_config
    ).to(device)
except Exception as e:
    print(f"Failed to load model with quantization: {e}")
    print("Falling back to CPU without quantization.")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Prepare model with Accelerator
model = accelerator.prepare(model)

# Verify model device
print(f"Model device: {next(model.parameters()).device}")

# Test tokenizer with sample input
sample_input = "Hello, how are you?"
inputs = tokenizer(sample_input, return_tensors="pt").to(device)
print(f"Tokenized input: {inputs}")

# Test model inference with explicit generation parameters
outputs = model.generate(
    **inputs,
    max_length=50,
    num_beams=5,
    length_penalty=0.65,
    no_repeat_ngram_size=3,
    do_sample=True,
    temperature=0.7
)
response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
print(f"Sample response: {response}")

# Save model and tokenizer locally
local_path = r"D:\AI\Models\blenderbot_400M"
tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)
print(f"Model and tokenizer saved to {local_path}")

print("Model and tokenizer loaded successfully!")