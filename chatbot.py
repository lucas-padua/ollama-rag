from openai import OpenAI
from src.rag_pipeline import RAGPipeline
import streamlit as st
import time

rag = RAGPipeline("Documents")


def response_stream(response: str):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.2)


with st.sidebar:
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Talk to a real doctor](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit endometriosis chatbot powered by LLama")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = str(rag.query(prompt))
    with st.chat_message("assistant"):
        st.write_stream(response_stream(msg))
    st.session_state.messages.append({"role": "assistant", "content": msg})
