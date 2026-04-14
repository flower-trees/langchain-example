from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_ollama import OllamaLLM


# 定义工具
def get_weather(location: str):
    return f"{location} 的天气是晴天，温度 25°C"

def get_time(location: str):
    return f"{location} 的当前时间是 12:00 PM"

weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="获取城市天气信息，输入城市名称"
)

time_tool = Tool(
    name="get_time",
    func=get_time,
    description="获取城市当前时间，输入城市名称"
)

# 初始化 Agent
llm = OllamaLLM(model="llama3:8b", temperature=0)

agent = initialize_agent(
    tools=[weather_tool, time_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 让 LLM 选择合适的工具
response = agent.run("告诉我上海当前时间")
print(response)
