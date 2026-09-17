import streamlit as st
import requests

# Page Configuration

st.set_page_config(
    page_title="HR Assistant",
    page_icon="💼",
    layout="centered"
)

# Title

st.title("💼 HR Assistant using LangGraph")
st.write("Ask questions about the HR policy document.")

# Session State

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input

prompt = st.chat_input("Ask your HR question...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    # Assistant
    with st.chat_message("assistant"):
        with st.spinner("Searching HR document..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"query": prompt},timeout=60
                )

                if response.status_code == 200:
                    answer = response.json()["answer"]
                else:
                    answer = response.json()["detail"]
            except Exception as e:
                answer = str(e)
            st.markdown(answer)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# Sidebar

with st.sidebar:
    st.title("Settings")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()