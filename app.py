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
#from dotenv import load_dotenv

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

#load_dotenv()

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
watsonx_project_id = "5bd59c57-bfa7-4565-b392-a98a90224509"#os.getenv("WATSONX_PROJECT_ID", None)
 
 

prompt_template_br = PromptTemplate(
    input_variables=["context", "question"], 
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
I am a helpful assistant.

<|eot_id|>
{context}
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
""")

with st.sidebar: 
    st.title("Watsonx RAG Demo")        
    #watsonx_api_key = st.text_input("Watsonx API Key", key="watsonx_api_key", value="CycH3S8_zauKHDxJvjtenKAkOnz6skxApg9VMECFyvX8",""""os.getenv("IBM_CLOUD_API_KEY"),"""" type="password")
    watsonx_api_key = st.text_input("Watsonx API Key", key="watsonx_api_key", value="CycH3S8_zauKHDxJvjtenKAkOnz6skxApg9VMECFyvX8", type="password")
    if not watsonx_project_id:
        watsonx_project_id = st.text_input("Watsonx Project ID", key="watsonx_project_id")
    watsonx_model = st.selectbox("Model", ["ibm/granite-20b-multilingual", "meta-llama/llama-3-405b-instruct"]) 
    max_new_tokens = st.slider("Max output tokens", min_value=100, max_value=900, value=300, step=100)
    decoding_method = st.sidebar.radio("Decoding", (DecodingMethods.GREEDY.value, DecodingMethods.SAMPLE.value))
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
        st.write("Filename:", uploaded_file.name)
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
params = parameters 
project_id = watsonx_project_id
space_id = None
verify = False
model = WatsonxLLM(model=Model(model_name, my_credentials, params, project_id, space_id, verify))

if model:
    st.info(f"Model {model_name} ready.")
    chain = LLMChain(llm=model, prompt=prompt_template_br, verbose=True)

if chain:
    st.info("Chat ready.")
    if index:
        rag_chain = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",
                retriever=index.vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": prompt_template_br},
                return_source_documents=False,
                verbose=True
            )
        st.info("Chat with PDF document ready.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Ask your question here", disabled=False if chain else True)

if prompt:
    st.chat_message("user").markdown(prompt)

    # Try to get response from document, if available
    response_text = None
    if rag_chain:
        response_text = rag_chain.run(prompt).strip()
    
    # If no response from document, fall back to LLM
    if not response_text:
        response = chain.run(question=prompt, context=format_history())
        response_text = response.strip("<|start_header_id|>assistant<|end_header_id|>").strip("<|eot_id|>")
        
    # Display response and log it in session history
    st.session_state.messages.append({'role': 'User', 'content': prompt })
    st.chat_message("assistant").markdown(response_text)
    st.session_state.messages.append({'role': 'Assistant', 'content': response_text })
