import getpass
import os

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 创建用于天气查询和活动推荐的模拟函数
def get_weather(location):
    """模拟天气查询工具函数"""
    # 假设返回晴朗天气
    return "晴朗，温度 25°C"

def suggest_outdoor_activities(weather):
    """基于天气推荐户外活动"""
    if "晴朗" in weather:
        return "推荐进行远足、骑行或野餐。"
    else:
        return "不推荐户外活动，建议在室内进行活动。"

# 将函数封装为 LangChain 工具
weather_tool = Tool(
    name="Weather Query",
    func=lambda location: get_weather(location),
    description="根据位置查询当前天气情况。"
)

activity_tool = Tool(
    name="Activity Suggestion",
    func=lambda weather: suggest_outdoor_activities(weather),
    description="根据天气情况推荐适合的活动。"
)

# 初始化语言模型 (可以使用 OpenAI 或其他 LLM)
llm = OllamaLLM(model="llama3:8b", temperature=0)

# 构建 Prompt 模板，用于指导模型逐步推理和行动
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
你是一个智能助手，用户向你查询户外活动的建议。
首先，查询用户位置的天气情况，然后基于天气推荐适合的活动。
如果天气适合户外活动，推荐户外活动；如果不适合，则建议室内活动。
用户的请求：{query}
"""
)

# 初始化代理，设置为 ReAct 风格
agent = initialize_agent(
    tools=[weather_tool, activity_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 设置为 ReAct 风格
    verbose=True
)

# 用户查询
query = "帮我查询纽约的天气，并推荐适合的活动。"

# 使用代理处理查询
response = agent({"input": query})

print(response["output"])
