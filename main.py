from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

import streamlit as st
import os
from dotenv import load_dotenv

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

load_dotenv()

@st.cache_resource
def load_pdf(pdf_name):
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
    ).from_loaders(loaders)
    return index

def format_history():
    text = ""
    return text

model = None
index = None
chain = None
rag_chain = None
watsonx_project_id = os.getenv("WATSONX_PROJECT_ID", None)

# Prompt Template for LLM in English
prompt_template = PromptTemplate(
    input_variables=["context", "question"], 
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
I am an assistant who always provides helpful, respectful, and honest responses. I will answer in a constructive, positive, and neutral way.

If a question is nonsensical or factually incorrect, explain why rather than providing an incorrect answer. If you don't know the answer, do not share false information.

<|eot_id|>
{context}
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
"""
)

with st.sidebar: 
    st.title("watsonx RAG Demo")        
    watsonx_api_key = st.text_input("Watsonx API Key", key="watsonx_api_key", value=os.getenv("IBM_CLOUD_API_KEY", None), type="password")
    if not watsonx_project_id:
        watsonx_project_id = st.text_input("Watsonx Project ID", key="watsonx_project_id")
    
    watsonx_model = st.selectbox("Model", ["ibm/granite-20b-multilingual", "meta-llama/llama-3-405b-instruct"]) 
    max_new_tokens = st.slider("Max output tokens", min_value=100, max_value=900, value=300, step=100)
    decoding_method = st.radio("Decoding", (DecodingMethods.GREEDY.value, DecodingMethods.SAMPLE.value))
    
    parameters = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE: 0,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1,
        GenParams.STOP_SEQUENCES: [],
        GenParams.REPETITION_PENALTY: 1
    }
    
    st.info("Upload a PDF file to use RAG")
    uploaded_file = st.file_uploader("Upload file", accept_multiple_files=False)
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        with open(uploaded_file.name, 'wb') as f:
            f.write(bytes_data)
        index = load_pdf(uploaded_file.name)

    model_name = watsonx_model
    
    def clear_messages():
        st.session_state.messages = []
        
    st.button('Clear messages', on_click=clear_messages)

st.info("Setting up watsonx...")

my_credentials = { 
    "url": "https://us-south.ml.cloud.ibm.com", 
    "apikey": watsonx_api_key
}
project_id = watsonx_project_id
space_id = None
verify = False

model = WatsonxLLM(model=Model(model_name, my_credentials, parameters, project_id, space_id, verify))

if model:
    st.info("Model {} is ready.".format(model_name))
    chain = LLMChain(llm=model, prompt=prompt_template, verbose=True)

if chain:
    st.info("Chat is ready.")
    if index:
        rag_chain = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",
                retriever=index.vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=False,
                verbose=True
            )
        st.info("Chat is ready with PDF document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Ask your question here", disabled=False if chain else True)

if prompt:
    st.chat_message("user").markdown(prompt)

    response = chain.run(question=prompt, context=format_history())
    response_text = response.lstrip("<|start_header_id|>assistant<|end_header_id|>").rstrip("<|eot_id|>")
    st.session_state.messages.append({'role': 'User', 'content': prompt})
    st.chat_message("assistant").markdown("[LLM] " + response_text)
    st.session_state.messages.append({'role': 'Assistant', 'content': "[LLM] " + response_text})

    if rag_chain:
        response = rag_chain.run(prompt)
        response_text = response.lstrip("<|start_header_id|>assistant<|end_header_id|>").rstrip("<|eot_id|>")
        st.chat_message("assistant").markdown("[DOC] " + response_text)
        st.session_state.messages.append({'role': 'Assistant', 'content': "[DOC] " + response_text})
