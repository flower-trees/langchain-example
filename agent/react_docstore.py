from langchain.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_ollama import ChatOllama

# 加载文档并创建向量存储
# loader = PyPDFLoader("knowledge.txt")
loader = PyPDFLoader("/Users/chuanzhizhu/Downloads/朱传志-A9.pdf")
docs = loader.load()
vectorstore = FAISS.from_documents(docs, OllamaEmbeddings(model="nomic-embed-text"))


# 定义检索工具
def retrieve_docs(query: str):
    return vectorstore.similarity_search(query, k=3)


docstore_tool = Tool.from_function(
    name="Lookup",
    func=lambda query: retrieve_docs(query),
    description="检索文档内容"
)


def dummy_search(query: str):
    return "朱传志是个好学生"

search_tool = Tool(
    name="Search",
    func=lambda query: dummy_search(query),
    description="模拟搜索工具，不执行实际搜索"
)


# 初始化代理
agent = initialize_agent(
    tools=[docstore_tool, search_tool],
    llm=ChatOllama(model="llama3:8b", temperature=0),
    agent=AgentType.REACT_DOCSTORE
)

agent.verbose = True
agent.handle_parsing_errors = True

# 运行查询
response = agent.run("文档中描述朱传志在美团经历有那些")
print(response)
