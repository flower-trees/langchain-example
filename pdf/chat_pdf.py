# 导入必要的库
import asyncio

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 设置PDF文件路径
pdf_path = "../files/pdf/en/Transformer.pdf"

# 1. 加载PDF文档
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 2. 切分文本
# 将文档分割为 1000 个字符的块，块之间有 200 个字符的重叠
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 3. 存储分割后的文档到向量数据库
# 使用 OpenAI 嵌入模型生成文档向量表示，并存储到 Chroma 向量数据库中
vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="nomic-embed-text"))

# 4. 构建检索器
# 使用相似度检索找到与查询最相关的文档
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 5. 定义 RAG 提示模板
prompt = hub.pull("rlm/rag-prompt")

# 格式化检索到的文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. 定义 RAG 链
# 创建检索增强生成链：包括检索、格式化文档、提示模板和语言模型生成答案
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOllama(model="deepseek-r1:7b")
    | StrOutputParser()
)

print("RAG ready")

# 7. 生成答案
# 传入问题并获取答案
# invoke
response = rag_chain.invoke("Why is masking necessary in the decoder’s self-attention mechanism?")

# 打印答案
print(response)

# astream
async def get_streaming_response(question: str):
    async for chunk in rag_chain.astream(question):
        print(chunk, end="", flush=True)

asyncio.run(get_streaming_response("Why is masking necessary in the decoder’s self-attention mechanism?"))

# 8. 清理向量数据库
vectorstore.delete_collection()
