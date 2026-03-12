from dotenv import load_dotenv
import os
import streamlit as st
import requests

# Load environment variables from .env
load_dotenv()

ENDPOINT = os.getenv("MODEL_ENDPOINT")
TIMEOUT = int(os.getenv("API_TIMEOUT", 30))

if not ENDPOINT:
    st.error("MODEL_ENDPOINT is not set in .env")
    st.stop()

# Page configuration
st.set_page_config(page_title="GPT-2 Joke Generator", page_icon="😂")

st.title("GPT-2 Joke Generator")
st.caption("Type something and GPT-2 will turn it into a joke.")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Show previous messages
for entry in st.session_state.history:
    st.chat_message("user").write(entry["prompt"])
    st.chat_message("assistant").write(entry["joke"])

# User input
if prompt := st.chat_input("Start a joke... e.g. Why did the programmer"):

    st.chat_message("user").write(prompt)

    with st.spinner("Generating joke..."):
        try:
            response = requests.post(
                ENDPOINT,
                json={"prompt": prompt},
                timeout=TIMEOUT
            )

            response.raise_for_status()

            data = response.json()
            joke = data.get("joke", "Model returned no joke.")

        except requests.exceptions.RequestException as e:
            joke = f"API request failed: {e}"
        except ValueError:
            joke = "Invalid JSON response from API."

    st.chat_message("assistant").write(joke)

    st.session_state.history.append({
        "prompt": prompt,
        "joke": joke
    })