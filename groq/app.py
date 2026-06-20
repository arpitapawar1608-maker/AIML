import gradio as gr
from groq import Groq

# Replace with your NEW Groq API key
client = Groq(
    api_key="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"
)

def get_response(prompt):

    if not prompt.strip():
        return "Please enter a prompt."

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(
        lines=5,
        label="Prompt",
        placeholder="Ask me anything..."
    ),
    outputs=gr.Textbox(
        lines=10,
        label="Response"
    ),
    title="🤖 Groq AI Assistant",
    description="Simple Gradio chatbot using Groq"
)

demo.launch()