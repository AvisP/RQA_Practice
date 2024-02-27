# import utils

import os
import openai
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])

from llama_index.core import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))


#### Windows sentence Retrival setup

from llama_index.core.node_parser import SentenceWindowNodeParser

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
llm = Ollama(model="llama2", request_timeout=120.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = node_parser

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import load_index_from_storage

sentence_index = VectorStoreIndex.from_documents(
    [document], service_context=Settings, show_progress=True
)

sentence_index.storage_context.persist(persist_dir="./sentence_index")

### Load if already existing

if not os.path.exists("./sentence_index"):
    sentence_index = VectorStoreIndex.from_documents(
        [document], service_context=Settings, show_progress=True
    )

    sentence_index.storage_context.persist(persist_dir="./sentence_index")
else:
    sentence_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./sentence_index"),
        service_context=Settings
    )

## Running the postprocessor
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)

from llama_index.core.schema import NodeWithScore
from copy import deepcopy

text = "hello. foo bar. cat dog. mouse"

nodes = node_parser.get_nodes_from_documents([Document(text=text)])

scored_nodes = [NodeWithScore(node=x, score=1.0) for x in nodes]
nodes_old = [deepcopy(n) for n in nodes]

print(scored_nodes)
print(nodes_old[1])

replaced_nodes = postproc.postprocess_nodes(scored_nodes)

print(replaced_nodes[1])

## Adding a reranker

from llama_index.core.postprocessor import SentenceTransformerRerank

# BAAI/bge-reranker-base
# link: https://huggingface.co/BAAI/bge-reranker-base
rerank = SentenceTransformerRerank(
    top_n=2, model="BAAI/bge-reranker-base"
)

from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.schema import TextNode, NodeWithScore

query = QueryBundle("I want a dog.")

scored_nodes = [
    NodeWithScore(node=TextNode(text="This is a cat"), score=0.6),
    NodeWithScore(node=TextNode(text="This is a dog"), score=0.4),
]

reranked_nodes = rerank.postprocess_nodes(
    scored_nodes, query_bundle=query
)

print([(x.text, x.score) for x in reranked_nodes])

### Running the query engine

sentence_window_engine = sentence_index.as_query_engine(
    similarity_top_k=6, node_postprocessors=[postproc, rerank]
)

window_response = sentence_window_engine.query(
    "What are the keys to building a career in AI?"
)

print(window_response)