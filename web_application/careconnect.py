import time
import streamlit as st
from rag import RAG
from model import LLM
import warnings

warnings.filterwarnings("ignore")

print('CareConnect Started.', flush=True)

@st.cache_resource(show_spinner=False)
def load_rag():
    """Load the RAG (Retrieval-Augmented Generation) database."""
    print('Loading the database.', flush=True)
    return RAG()

@st.cache_resource(show_spinner=False)
def load_llm():
    """Load the LLM (Large Language Model)."""
    print('Loading the model.', flush=True)
    return LLM()

try:
    rag = load_rag()
    llm = load_llm()
    print('CareConnect is ready!', flush=True)
except Exception as e:
    print('Something went wrong!', flush=True)
    print(e)
    exit()

def get_response(query):
    """Get the response from the LLM based on the query and context from RAG.

    Args:
        query (str): The query to be processed.

    Returns:
        str: The response from the LLM or an error message.
    """
    context = rag.search(query)
    response = llm.inference(query, context)
    return response if response is not None else "Something went wrong! Please try again later."

# Show title and description.
st.image("careConnectCroppped.png", width=200)
st.title("CareConnect")
st.write("Welcome to CareConnect! Your personal health assistant!")

# Initialize chat messages in session state if not already present.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi I'm CareConnect! How can I help?"}]

# Display existing chat messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input field for the user to enter messages.
if prompt := st.chat_input("What is up?"):
    # Store and display the user's prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the response from the LLM.
    start = time.perf_counter()
    response = get_response(prompt)
    print(f'Response time: {time.perf_counter() - start} seconds.', flush=True)
    
    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
