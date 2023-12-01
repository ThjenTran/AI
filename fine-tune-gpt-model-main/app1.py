from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI 
from jina import Document
import gradio as gr
import os

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 2049 #4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-ada-001", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex.from_documents(documents) 

    index.save_to_disk('index.json')

    return os.path.abspath('index.json')

def chatbot(input_text):     
    index = GPTSimpleVectorIndex.load_from_disk('index.json')         
    response = index.query(input_text, response_mode="compact")    
    return response.response
                       
                   
index_path = construct_index("docs")

iface = gr.Interface(fn=chatbot,
                    inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                    outputs="text",
                    title="Custom-trained AI Chatbot")
                    
iface.launch(share=True)

#wget https://example.com/file.zip
