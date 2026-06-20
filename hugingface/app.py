import streamlit as st
from huggingface_hub import InferenceClient


st.title("🤗 Hugging Face Chatbot")


HF_TOKEN = st.secrets["HF_TOKEN"]


client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)


prompt = st.text_input("Ask something")


if prompt:

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500
    )


    answer = response.choices[0].message.content

    st.write(answer)