from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
import asyncio
import nest_asyncio
import random

nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader

llm = Ollama(model="llama2", request_timeout=1200.0)
documents = SimpleDirectoryReader(
    input_files=["./IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

text_splitter = SentenceSplitter()
base_nodes = text_splitter.get_nodes_from_documents(documents)

num_nodes_eval = 30
# there are 428 nodes total. Take the first 200 to generate questions (the back half of the doc is all references)
sample_eval_nodes = random.sample(base_nodes[:200], num_nodes_eval)
# NOTE: run this if the dataset isn't already saved
# generate questions from the largest chunks (1024)
dataset_generator = DatasetGenerator(
    sample_eval_nodes,
    llm=llm,
    show_progress=True,
    num_questions_per_chunk=2,
)

loop = asyncio.get_event_loop()
eval_dataset = loop.run_until_complete(dataset_generator.agenerate_dataset_from_nodes())
loop.close()

eval_dataset.save_json("data/ipcc_eval_qr_dataset.json")
# optional
eval_dataset = QueryResponseDataset.from_json("data/ipcc_eval_qr_dataset.json")

