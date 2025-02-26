import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from starlette.responses import StreamingResponse

# 设置 PDF 文件路径
pdf_path = "../files/pdf/en/Transformer.pdf"

# 加载 PDF 文档并分割文本
loader = PyPDFLoader(pdf_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# 存储分割后的文档到向量数据库
vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="nomic-embed-text"))

# 构建检索器
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 定义 RAG 提示模板
prompt = hub.pull("rlm/rag-prompt")


# 格式化检索到的文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 定义 RAG 链
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOllama(model="deepseek-r1:7b")
        | StrOutputParser()
)

print("RAG ready")


# 生成答案函数
async def generate_answer(question: str):
    response = await rag_chain.ainvoke(question)
    return response

# 生成流式响应
async def generate_streaming_response(question: str):
    async for chunk in rag_chain.astream(question):  # 使用 astream 逐块获取响应
        yield json.dumps({"answer chunk": chunk}) + "\n"  # 按流式返回每一块内容

# 8. 清理向量数据库
def clear_vectorstore():
    vectorstore.delete_collection()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在应用启动时执行的代码
    yield
    # 在应用关闭时执行的代码
    clear_vectorstore()
    print("Vectorstore cleaned up successfully!")

# 创建 FastAPI 应用
app = FastAPI(lifespan=lifespan)

# 定义输入模型
class QueryModel(BaseModel):
    question: str
    stream: bool = False  # 默认不流式返回

# 创建 POST 路由处理查询
@app.post("/query/")
async def query_question(query: QueryModel):
    try:
        if query.stream:
            # 如果 `stream` 为 True，使用流式响应
            return StreamingResponse(generate_streaming_response(query.question), media_type="text/json")
        else:
            # 否则直接返回完整答案
            answer = await generate_answer(query.question)  # 使用 await 获取完整的答案
            return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动 FastAPI 应用（适用于开发环境）
# uvicorn chat_pdf_api:app --reload
