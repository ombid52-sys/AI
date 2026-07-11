import streamlit as st
import openai
from google import genai
from google.genai import errors

st.set_page_config(page_title="Universal AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Multi-Provider AI Chatbot")
st.caption("Enter an OpenAI or Gemini API key, and the app will auto-detect the provider and start the chat.")

# Initialize chat history and provider status in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "provider" not in st.session_state:
    st.session_state.provider = None

# Sidebar for API Key Input
with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("Enter AI API Key:", type="password", help="Supports OpenAI (sk-...) or Gemini keys")
    
    # Button to validate and detect key provider
    if st.button("Detect & Verify Provider"):
        if not api_key.strip():
            st.error("Please enter a valid API key.")
        else:
            with st.spinner("Analyzing key signature..."):
                detected_provider = None
                
                # Rule 1: Structural hints
                if api_key.startswith("sk-"):
                    detected_provider = "OpenAI"
                else:
                    # Let's test if it works as a Gemini key
                    try:
                        client = genai.Client(api_key=api_key)
                        # Quick validation call
                        client.models.list(config={'page_size': 1})
                        detected_provider = "Gemini"
                    except Exception:
                        # If Gemini check fails and it wasn't explicitly OpenAI structure, check OpenAI fallback
                        try:
                            client = openai.OpenAI(api_key=api_key)
                            client.models.list()
                            detected_provider = "OpenAI"
                        except Exception:
                            detected_provider = None

                if detected_provider == "OpenAI":
                    st.session_state.provider = "OpenAI"
                    st.session_state.api_key = api_key
                    st.success("✨ Successfully detected OpenAI Key!")
                elif detected_provider == "Gemini":
                    st.session_state.provider = "Gemini"
                    st.session_state.api_key = api_key
                    st.success("✨ Successfully detected Google Gemini Key!")
                else:
                    st.error("❌ Invalid Key or Unrecognized Provider. Please check your token.")
                    st.session_state.provider = None

    if st.session_state.provider:
        st.info(f"Active Session: **{st.session_state.provider}**")
        if st.button("Clear Chat / Change Key"):
            st.session_state.messages = []
            st.session_state.provider = None
            st.rerun()

# --- Chat Interface Layout ---
# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user interaction if provider is active
if st.session_state.provider:
    if prompt := st.chat_input("Say something to your AI..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response based on detected provider
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                if st.session_state.provider == "OpenAI":
                    client = openai.OpenAI(api_key=st.session_state.api_key)
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                            
                elif st.session_state.provider == "Gemini":
                    client = genai.Client(api_key=st.session_state.api_key)
                    # Convert simple message history format to Gemini format if necessary, 
                    # or just pass the text for a single turn context
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt
                    )
                    full_response = response.text
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"An error occurred while calling the API: {e}")
else:
    st.warning("👈 Please enter and verify your API key in the sidebar to open up the chat line.")