#from gpt_index import GPTSimpleVectorIndex
from llama_index import SimpleDirectoryReader, GPTListIndex, LLMPredictor, PromptHelper, ServiceContext, GPTVectorStoreIndex
from llama_index import StorageContext
from llama_index.node_parser import SimpleNodeParser

from llama_index import (
    GPTVectorStoreIndex,
    ResponseSynthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index import StorageContext, load_index_from_storage

from langchain import OpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'
    
def construct_index(directory_path):
    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003"))

    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    chunk_size_limit = 600
    
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap,chunk_size_limit=chunk_size_limit)
    
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    #parser = SimpleNodeParser()

    #nodes = parser.get_nodes_from_documents(documents)

    #index = GPTVectorStoreIndex(nodes)
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
    #index = GPTVectorStoreIndex.from_documents(documents)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    
    #storage_context = StorageContext.from_defaults(persist_dir=directory_path)
    
    index.storage_context.persist(persist_dir="./index.json")
        
    #return index

def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir="index.json")
    storage_context1 = StorageContext.from_defaults(persist_dir="index1.json")
    storage_context.docstore.add_documents(storage_context1.docstore.document_exists)
    storage_context.index_store.add_index_struct(storage_context1.index_store.index_structs)
    storage_context.vector_store.add(storage_context1.vector_store)
    
    index = load_index_from_storage(storage_context) 
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
     
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

#index = construct_index("docs")
 
iface.launch(share=True)