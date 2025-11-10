import gradio as gr
import os
from main import get_setup_advice  # import your existing RAG chain function

def answer_question(question, telemetry_file):
    """Handle driver question + optional telemetry file."""
    file_path = None
    if telemetry_file is not None:
        file_path = telemetry_file.name
    return get_setup_advice(question, file_path)

# Define Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(
            label="Driver Question",
            placeholder="e.g. My dirt midget is loose on entry",
            lines=2
        ),
        gr.File(label="Upload Telemetry CSV (optional)")
    ],
    outputs=gr.Textbox(label="Setup Advice", lines=10),
    title="🏁 iRacing Setup Assistant",
    description=(
        "Ask for setup help and get short, specific tuning suggestions "
        "(e.g., 'Add 0.5 rebound to RR shock'). Make sure Ollama is running."
    ),
    theme="gradio/soft"
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)
