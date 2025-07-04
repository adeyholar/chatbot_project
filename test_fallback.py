from chatbot_api import ChatbotAPI
from dotenv import load_dotenv
import os

def test_fallback():
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY not found in .env")

    # Test with invalid API URL to force fallback
    invalid_url = "https://api-inference.huggingface.co/models/invalid-model"
    chatbot = ChatbotAPI(api_key, api_url=invalid_url, local_model_path=r"D:\AI\Models\blenderbot_1B")
    try:
        response = chatbot.generate_response("Hello, how are you?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Test with valid API URL
    valid_url = "https://api-inference.huggingface.co/models/t5-small"
    chatbot = ChatbotAPI(api_key, api_url=valid_url, local_model_path=r"D:\AI\Models\blenderbot_1B")
    try:
        response = chatbot.generate_response("Hello, how are you?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fallback()