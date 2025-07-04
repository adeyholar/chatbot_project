from chatbot import Chatbot
import gradio as gr
import os

# Set Hugging Face cache directory
os.environ["HF_HOME"] = r"D:\AI\Models\huggingface"

def chat_interface(user_input, history=None):
    # Initialize history if None
    if history is None:
        history = []

    # Load chatbot
    model_path = r"D:\AI\Models\blenderbot_400M"
    chatbot = Chatbot(model_path)

    # Generate response
    response = chatbot.generate_response(user_input)

    # Update history
    history.append((user_input, response))
    return history, history

# Create Gradio interface
iface = gr.Interface(
    fn=chat_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Type your message here..."),
        gr.State()
    ],
    outputs=[
        gr.Chatbot(label="Conversation"),
        gr.State()
    ],
    title="BlenderBot Chatbot",
    description="Chat with a local BlenderBot-400M model. Type 'exit' in the console to stop the server."
)

# Launch interface
if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)