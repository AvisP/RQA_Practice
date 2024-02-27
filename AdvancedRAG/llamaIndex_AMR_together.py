import os
# from llama_index.core import Settings, VectorStoreIndex, StorageContext
# from llama_index.core.node_parser import SentenceWindowNodeParser
# from llama_index.core.postprocessor import MetadataReplacementPostProcessor
# from llama_index.core.postprocessor import SentenceTransformerRerank
# from llama_index.core.indices.loading import load_index_from_storage

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

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

    # merging_context = ServiceContext.from_defaults(
    #     llm=llm,
    #     embed_model=embed_model,
    # )
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
llm = Ollama(model="llama2", request_timeout=1200.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

# resp = llm.complete("Who is Paul Graham?")
# print(resp)
print("Model loading finished")

from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

document = Document(text="\n\n".join([doc.text for doc in documents]))

print("Document loading finished")

index = build_automerging_index(
    [document],
    llm=llm,
    embed_model=embed_model,
    save_dir="./merging_index",
)
print("Index building complete")

query_engine = get_automerging_query_engine(index, similarity_top_k=6)

window_response = query_engine.query(
    "What are the keys to building a career in AI?"
)

print(window_response)