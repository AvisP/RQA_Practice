## Streamlit example of doing RAG using Auto Merging Retrival

import os
import streamlit as st
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
# from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.loading import load_index_from_storage

from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

def build_sentence_window_index(
    documents,
    llm,
    embed_model,
    sentence_window_size=3,
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=Settings, show_progress=True
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=Settings,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine

def build_automerging_index(
    documents,
    llm,
    embed_model="BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, 
            storage_context=storage_context, 
            service_context=Settings,
            show_progress=True
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=Settings,
            show_progress=True
        )

    return automerging_index

def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document

st.title("🦙 Llama Index Advanced RAG Demo 🦙")
st.header("Welcome to the Llama Index Streamlit Demo")

loaded_model = "llama2"
llm = Ollama(model=loaded_model, request_timeout=1200.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)

text = st.text_input("Query text:", value="What are the keys to building a career in AI?")

with st.sidebar:
    option = st.selectbox(
        "How would you like to perform retrival?",
        ("Sentence Window Retrival", "Auto Merging Retrival"),
        index=None,
        placeholder="Select Advanced RAG Method...",
        )
    
    select_llm_model = st.selectbox(
        "Which llm model you would like to use?",
        ("llama2:7b","gemma:7b","mistral:7b"),
        index=None,
        placeholder="Select LLama model...",
        )
    
    temperature_value = st.slider(
        'Temperature',
        min_value=0.0,
        max_value=1.0, 
        value=0.1,
        step=0.01)
    
if temperature_value != llm.temperature:
    llm = Ollama(model=loaded_model, temperature=temperature_value, request_timeout=1200.0)
    
if select_llm_model != loaded_model and select_llm_model is not None:
    with st.spinner(text="Downloading and loading selected model"):
        st.write("Selected model is :", select_llm_model)
        loaded_model = select_llm_model
        llm = Ollama(model=loaded_model, temperature=temperature_value, request_timeout=1200.0)

        print("Loading of model " + loaded_model + " finished")

if uploaded_file is not None:
    with st.spinner(text="Loading and indexing the uploaded data – hang tight! This should take 1-2 minutes."):
        documents = SimpleDirectoryReader(
            # input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
            input_files = [uploaded_file.name]
        ).load_data()

        document = Document(text="\n\n".join([doc.text for doc in documents]))

        print("Document loading finished")

if option is not None and uploaded_file is not None:
    if option == "Sentence Window Retrival":
        st.write("Selected technique:", option)
        index = build_sentence_window_index(
                [document],
                llm=llm,
                embed_model=embed_model,
                save_dir="./sentence_index",
            )

        query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)
    elif option == "Auto Merging Retrival":
        st.write("Selected technique:", option)
        index = build_automerging_index(
                [document],
                llm=llm,
                embed_model=embed_model,
                save_dir="./merging_index",
            )

        query_engine = get_automerging_query_engine(index, similarity_top_k=6)


if st.button("Run Query") and text is not None:
    window_response = query_engine.query(text)

    print(window_response)
    st.markdown(window_response)