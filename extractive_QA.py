from transformers import pipeline
from duckduckgo_search import DDGS
import torch
import streamlit as st
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIPELINE_NAME = "question-answering"
# MODEL_NAME = "binhphap5/distilbert-finetuned-squadv2"

def load_model(model_name):
    """
    Load the model for question answering

    Returns
    -------
    pipeline
        The model for question answering
    """
    pipe = pipeline(PIPELINE_NAME, model=model_name, device=device)
    return pipe

def get_context_from_duckduckgo(question, max_results):
    """
    Search for the given question using duckduckgo and return the snippets joined as the context

    Parameters
    ----------
    question : str
        The question to search for
    max_results : int
        The maximum number of results to retrieve

    Returns
    -------
    str
        The context to query with the model
    """
    with DDGS() as ddgs:
        results = ddgs.text(question, max_results=max_results)
        snippets = [result['body'] for result in results]
        href = [result['href'] for result in results]
        context = " ".join(snippets)
    return context, href

def get_answer(question, context, model_name):
    """
    Get the answer to the question using the model

    Parameters
    ----------
    question : str
        The question to ask

    model_name : str
        The name of the model to use

    Returns
    -------
    str
        The answer to the question
    """
    pipe = load_model(model_name)
    return pipe(question=question, context=context)

#--------------------------- Streamlit UI ----------------------------------
st.title("Extractive Question Answering")
question = st.text_input("Enter your question")
model_name = st.selectbox("Question Answering Model", ["binhphap5/distilbert-finetuned-squadv2"], placeholder="Select a model")
max_contexts = st.number_input("Enter the maximum number of contexts to retrieve", min_value=1, max_value=20, value=2)
if st.button("Submit"):
    if question == "":
        st.write("Please enter a question")
    elif model_name == "":
        st.write("Please select a model")
    else:
        context, ref = get_context_from_duckduckgo(question, max_results=max_contexts)
        st.write("References:")
        st.write(ref)

        st.write("")

        st.write("Retrieved Context:")
        st.write(context)
        
        st.write("")

        answer = get_answer(question, context, model_name)
        st.write("Result:")
        st.write(answer)