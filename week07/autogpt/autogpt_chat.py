import gradio as gr
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

class ModelConfig:
    def __init__(self, name, title):
        self.name = name
        self.title = title

MODEL_CONFIGS = [
    ModelConfig("gpt-3.5-turbo", "GPT3.5"),
    ModelConfig("gpt-4", "GPT4"),
]


def new_tools():
    search = SerpAPIWrapper()
    return [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        WriteFileTool(),
    ]

def new_vectorstore():
    embeddings_model = OpenAIEmbeddings()
    # OpenAI Embedding 向量维数
    embedding_size = 1536
    # 使用 Faiss 的 IndexFlatL2 索引
    index = faiss.IndexFlatL2(embedding_size)
    # 实例化 Faiss 向量数据库
    return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


def initialize_models(models):
    tools = new_tools()
    vectorstore = new_vectorstore()
    global CHAT_MODELS
    CHAT_MODELS = {}
    for m in models:
        CHAT_MODELS[m.name] = AutoGPT.from_llm_and_tools(
            ai_name="Jarvis",
            ai_role="Assistant",
            tools=tools,
            llm=ChatOpenAI(model_name=m.name, temperature=0),
            memory=vectorstore.as_retriever(), # 实例化 Faiss 的 VectorStoreRetriever
        )

def get_chat_fn(model_name):
    def chat_fn(message, history):
        model = CHAT_MODELS[model_name]
        return model.run(message.split("\n"))
    return chat_fn


def launch_gradio(models):
    tab_list = []
    tab_names = []
    for m in models:
        tab =  gr.ChatInterface(
            fn=get_chat_fn(m.name),
            chatbot=gr.Chatbot(height=600),
        )
        tab_list.append(tab)
        tab_names.append(m.title)

    demo = gr.TabbedInterface(tab_list, tab_names)
    demo.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    initialize_models(MODEL_CONFIGS)
    launch_gradio(MODEL_CONFIGS)