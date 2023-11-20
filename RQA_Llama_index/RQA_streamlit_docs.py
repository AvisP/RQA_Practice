import streamlit as st
from htmlTemplates import css, bot_template, user_template
import os
import shutil
import json
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
import torch
from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index import set_global_tokenizer
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt


@st.cache_resource
def load_llm(temperature):

    llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    # model_path="/Users/paula/Projects/Text Gen/Llama2/models/Llama-2-7B-GGUF/llama-2-7b.Q8_0.gguf",
    model_path = "/Users/paula/Projects/Text Gen/Llama2/models/Mistral-7B-GGUF/mistral-7b-v0.1.Q8_0.gguf",
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 2},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)
    return llm

def init_query_engine(index, service_context, similarity_top_k=4, similarity_cutoff=0.7):
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=int(similarity_top_k),
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(service_context=service_context)

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=float(similarity_cutoff))],
    )

    return query_engine

def generate_index_varibale(service_context, document_directory="./PDFData/"):

    if not os.path.exists("./storage"):
        # load the documents and create the index
        documents = SimpleDirectoryReader(document_directory).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        # store it for later
        index.storage_context.persist()
    else:
        # load the existing index
        dir = os.listdir("./storage")
        if len(dir) == 0:
            st.error("Storage folder is empty, attempting to create from PDF directory")

            try:
                documents = SimpleDirectoryReader(document_directory).load_data()
                index = VectorStoreIndex.from_documents(documents, service_context=service_context)
                # store it for later
                index.storage_context.persist()
                st.warning("created embedding database from PDF directory")
            except:
                st.error("failed to create from PDF directory")

        else:
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            index = load_index_from_storage(storage_context, service_context=service_context)

    return index

def handle_userinput(query_engine, user_question):
    #What is fibromyalgia?
    st.write(user_template.replace(
                "{{MSG}}", user_question), unsafe_allow_html=True)
    response =  query_engine.query(user_question) #query_engine.query(user_question)
    response_str = response.__str__()
    file_path_index = response_str.find('file_path')
    response_str_file_path = response_str[file_path_index:]
    actual_response_index = response_str_file_path.find('\n\n')
    actual_response = response_str_file_path[actual_response_index:]

    if file_path_index == -1 or actual_response_index == -1:
        response_to_display = response_str
    else:
        response_to_display = actual_response
    st.session_state.chat_history = response
    print(response)
    st.write(bot_template.replace(
                "{{MSG}}", response_to_display), unsafe_allow_html=True)

def filepath_and_config_checks():

    if not os.path.exists('config.json'):
        st.error("Config File missing")

    config_file = open('config.json')
    config_data =json.loads(config_file.read())

    if not os.path.exists(config_data["llm_model_path"]):
        st.error("LLama model file path not found")

    # if not os.path.exists(config_data["embedding_path"]):
    #     st.error("Embedding model path not found")

    # if not os.path.exists(config_data["dbDirectory"]):
    #     st.error("Vector Database path not found")

    if not os.path.exists(config_data["PDFDirectory"]):
        st.error("PDF Directory not found")

    return config_data

def get_dbList(dbDirectory):
    dbList = []

    for db in os.listdir(dbDirectory):
        file_list = os.listdir(os.path.join(dbDirectory, db))
        for file in file_list:
            if file.endswith(".faiss"):
                dbList.extend([db])

    return dbList

######### Main Function ############

def main():
    global config_data
    config_data = filepath_and_config_checks()
    # index = False
    pdfList = [file for file in os.listdir(config_data["PDFDirectory"]) if file.endswith(".pdf")]

    #load_dotenv()
    st.set_page_config(page_title="Talk to your documents",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Talk to your documents 📄")

    embed_model = LangchainEmbedding(
                    HuggingFaceEmbeddings(model_name="thenlper/gte-large")
                )
    llm = load_llm(0.01)

    service_context = ServiceContext.from_defaults(
                        chunk_size=256,
                        llm=llm,
                        embed_model=embed_model)
    
    # st.session_state.conversation = query_engine
    user_question = st.chat_input("Ask me quesions about document :")
    
    if user_question:

        index = generate_index_varibale(service_context)
        if index:
            query_engine = init_query_engine(index, service_context)

            handle_userinput(query_engine, user_question)


    with st.sidebar:
        ## Select PDFs and gnerate new Embedding Database
        st.subheader("Generate Vector Database")
        selected_pdfs = st.sidebar.multiselect("What document would I add?", pdfList)
        # db_name = st.sidebar.text_input("Name of the new Database from selected PDFs:")

        if st.button("Process", key="1"):
            with st.spinner("Processing"):
                # check if storage already exists
                index = generate_index_varibale(service_context)

        if st.button("Delete database", key="2"):
            if os.path.exists("./storage"):
                shutil.rmtree("./storage")
            else:
                st.error("Storage folder not found")

         ## Tune the temperature parameter 
        st.subheader("Temperature Parameter Selector")
        temperature_parameter = st.sidebar.slider(
            "Temperature Parameter",
            min_value=0.0,  # Minimum value
            max_value=2.0,  # Maximum value
            value=0.01, # Default value
            step=0.01  # Step size
        )

        chunk_size = st.sidebar.slider(
            "Chunk size",
            min_value=16,  # Minimum value
            max_value=2038,  # Maximum value
            value=256, # Default value
            step=8  # Step size
        )

        similarity_top_k = st.sidebar.slider(
            "Similarity Top k",
            min_value=2,  # Minimum value
            max_value=20,  # Maximum value
            value=10, # Default value
            step=1  # Step size
        )

        similarity_cutoff = st.sidebar.slider(
            "Similarity Cutoff",
            min_value=0.1,  # Minimum value
            max_value=1.0,  # Maximum value
            value=0.7, # Default value
            step=0.05  # Step size
        )

        if st.button("Apply settings", key="3"):
            llm = load_llm(temperature_parameter)
            service_context = ServiceContext.from_defaults(
                        chunk_size=int(chunk_size),
                        llm=llm,
                        embed_model=embed_model)
            try:
                query_engine = init_query_engine(index, service_context, similarity_top_k=similarity_top_k, similarity_cutoff=similarity_cutoff)
            except:
                st.error("Select Document and process first")

if __name__ == '__main__':
    main()