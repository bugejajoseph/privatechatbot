from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext
from llama_index import StorageContext, load_index_from_storage
from langchain import OpenAI
import gradio as gr
import os

os.environ["OPENAI_API_KEY"] = '' # replace this your OpenAI API key

def construct_index(directory_path):
    num_outputs = 512
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    

    docs = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    index.storage_context.persist()

    return index

def chatbot(input_text):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir='./storage')

    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

index = construct_index("docs")
iface.launch(share=True)