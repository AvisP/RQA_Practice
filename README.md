# RQA_Practice
A repo where I try out Retrival Question Answer (RQA) examples presented in different libraries (LangChain, Llama_index) using scripts or a Graphical User Interface. Anyone interested can use and modify the scripts as needed.

# RQA Llama Index

This repo contains a script as well as a GUI developed using streamlit where the user can load up a PDF file, create an embedding database, ask questions and retrieve response using provided llama_index classes. The GUI can be started using `streamlit run RQA_streamlit_docs.py`. It should work with any GGUF model, has been tested only with LLama-7B, Mistral-7B. The model can be automatically downloaded by setting it with required huggingface url under `load_llm` function in `model_url` or if it has already been downloaded locally then providing it in `model_path`. A sample PDF file for testing has also been included.

![screenshot](./RQA_Llama_index/images/GUI_Screenshot.png)

## To Do
- [ ] Add option for selecting llm model through GUI
- [ ] Add option to see number of tokens used
- [ ] Investigate RQA cause of RQA mechanism not working proprerly in some cases

### Extra Scripts
* `llama_index_test2.py` can be executed if GUI is not needed
* `llama_index_test.py` shows a RQA example without using RetrieverQueryEngine but it is not working properly. However it has an example of how to use tokens
