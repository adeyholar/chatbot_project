import requests
import hmac
import hashlib
from dotenv import load_dotenv
import os

load_dotenv()
MCP_SECRET = os.getenv("MCP_SECRET")
if not MCP_SECRET:
    raise ValueError("MCP_SECRET not found or empty in .env")

def generate_signature(message: str) -> str:
    if not isinstance(message, str):
        raise ValueError("Message must be a string")
    return hmac.new(
        MCP_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()

def test_mcp():
    url = "http://127.0.0.1:8000/mcp/inference"
    message = "Hello, how are you?"
    signature = generate_signature(message)
    payload = {"message": message, "signature": signature}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_mcp()