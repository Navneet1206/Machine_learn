import streamlit as st
from scripts.chatbot import BusinessModelChatbot

st.set_page_config(page_title="Business Model Chatbot", page_icon="💼")

bot = BusinessModelChatbot()

st.title("🤖 Business Model Chatbot")
st.markdown("Ask about business models, revenue strategies, value propositions, etc.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", key="input")
if st.button("Ask"):
    if user_input:
        response = bot.get_response(user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", response))

for speaker, msg in reversed(st.session_state.history):
    st.markdown(f"**{speaker}:** {msg}")
