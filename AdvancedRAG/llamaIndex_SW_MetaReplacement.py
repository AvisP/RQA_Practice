## https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo.html
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
llm = Ollama(model="llama2", request_timeout=1200.0)
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# base node parser is a sentence splitter
text_splitter = SentenceSplitter()

# llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
)

from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

nodes = node_parser.get_nodes_from_documents(documents)
base_nodes = text_splitter.get_nodes_from_documents(documents)

from llama_index.core import VectorStoreIndex

sentence_index = VectorStoreIndex(nodes)
base_index = VectorStoreIndex(base_nodes)

from llama_index.core.postprocessor import MetadataReplacementPostProcessor

query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
window_response = query_engine.query(
    "What are the concerns surrounding the AMOC?"
)
print(window_response)

window = window_response.source_nodes[0].node.metadata["window"]
sentence = window_response.source_nodes[0].node.metadata["original_text"]

print(f"Window: {window}")
print("------------------")
print(f"Original Sentence: {sentence}")

print("Normal VectorStoreIndex response")

query_engine = base_index.as_query_engine(similarity_top_k=5)
vector_response = query_engine.query(
    "What are the concerns surrounding the AMOC?"
)
print(vector_response)

for source_node in window_response.source_nodes:
    print(source_node.node.metadata["original_text"])
    print("--------")

for node in vector_response.source_nodes:
    print("AMOC mentioned?", "AMOC" in node.node.text)
    print("--------")

## Evaluation