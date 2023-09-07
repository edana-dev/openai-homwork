import gradio as gr
import yaml


from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

class SalesBot:
    def __init__(self, name: str, title: str, vector_store_dir: str, score_threshold=0.8):
        self.name = name
        self.title = title
        self.vector_store_dir = vector_store_dir
        self.score_threshold = score_threshold

def initialize_sales_bots(bots):
    global CHAT_BOTS
    CHAT_BOTS = {}
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    for b in bots:
        print(f"initialize {b.name} with {b.vector_store_dir}")
        db = FAISS.load_local(b.vector_store_dir, OpenAIEmbeddings())
        bot = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": b.score_threshold}))
        bot.return_source_documents = True
        CHAT_BOTS[b.name] = bot

def sales_chat_func(name):
    def sales_chat(message, history):
        print(f"[message]{message}")
        print(f"[history]{history}")
        # TODO: 从命令行参数中获取
        enable_chat = True

        ans = CHAT_BOTS[name]({"query": message})
        print(ans)
        # 如果检索出结果，或者开了大模型聊天模式
        # 返回 RetrievalQA combine_documents_chain 整合的结果
        if ans["source_documents"] or enable_chat:
            print(f"[result]{ans['result']}")
            print(f"[source_documents]{ans['source_documents']}")
            return ans["result"]
        # 否则输出套路话术
        else:
            return "这个问题我要问问领导"
    return sales_chat


def launch_gradio(bots):
    tab_list = []
    tab_names = []
    for b in bots:
        tab =  gr.ChatInterface(
            fn=sales_chat_func(b.name),
            chatbot=gr.Chatbot(height=600),
        )
        tab_list.append(tab)
        tab_names.append(b.title)

    demo = gr.TabbedInterface(tab_list, tab_names)
    demo.launch(share=False, inbrowser=True, server_name="0.0.0.0")

def load_sales_bots(path="conf.yaml"):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
        result = []
        for c in data['bots']:
            name = c['name']
            title = c['title']
            vector_store_dir = c['vector_store_dir']
            score_threshold = c['score_threshold'] if 'score_threshold' in c else 0.5
            result.append(SalesBot(name, title, vector_store_dir, score_threshold))            
        return result

if __name__ == "__main__":
    bots = load_sales_bots()
    # 初始化所有机器人
    initialize_sales_bots(bots)
    # 启动 Gradio 服务
    launch_gradio(bots)
