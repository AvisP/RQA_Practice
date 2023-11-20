import logging
import sys
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage

import torch
from llama_index import set_global_tokenizer

# huggingface
from transformers import AutoTokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").encode
)


from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
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

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
# from llama_index.service_context import ServiceContext

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)

token_counter = TokenCountingHandler(
    tokenizer=AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").encode
)

callback_manager = CallbackManager([token_counter])

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    callback_manager=callback_manager,
    llm=llm,
    # embed_model="local"
    embed_model=embed_model
)

# documents = SimpleDirectoryReader("./PDFData/").load_data()
# index = VectorStoreIndex.from_documents(documents, service_context=service_context)

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


token_counter.reset_counts()

query_engine = index.as_query_engine()
response = query_engine.query("What is Fibromyalgia?")

print(response)

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)

# print(response.source_nodes[0].text)
# print(response.source_nodes[1].text)