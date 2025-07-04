from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from chatbot_api import ChatbotAPI
import hmac
import hashlib
import os
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
MCP_SECRET = os.getenv("MCP_SECRET")
if not API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found in .env")
if not MCP_SECRET:
    raise ValueError("MCP_SECRET not found or empty in .env")

class MCPRequest(BaseModel):
    message: str
    signature: str

def verify_signature(message: str, signature: str) -> bool:
    if not isinstance(message, str) or not isinstance(signature, str):
        return False
    computed = hmac.new(
        MCP_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, signature)

@app.post("/mcp/inference")
async def mcp_inference(request: MCPRequest):
    if not verify_signature(request.message, request.signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    chatbot = ChatbotAPI(
        api_key=API_KEY,
        local_model_path=r"D:\AI\Models\blenderbot_1B"
    )
    try:
        response = chatbot.generate_response(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)