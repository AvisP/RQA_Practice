from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
# https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a.html
import logging
import sys
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage

import torch

from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    # model_path="/....../Llama2/models/Llama-2-7B-GGUF/llama-2-7b.Q8_0.gguf",
    model_path = "/...../Llama2/models/Mistral-7B-GGUF/mistral-7b-v0.1.Q8_0.gguf",
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

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
# from llama_index.service_context import ServiceContext

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model)

# check if storage already exists
if not os.path.exists("./storage"):
    # load the documents and create the index
    documents = SimpleDirectoryReader("./PDFData/").load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # store it for later
    index.storage_context.persist()
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context, service_context=service_context)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(service_context=service_context)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# query
# response = query_engine.query("What are treatments for fibromylagia")
response = query_engine.query("What is fibromylagia")
print(response)
