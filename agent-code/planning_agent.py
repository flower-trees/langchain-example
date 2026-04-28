"""
规划 Agent - LangChain 实现
职责：读取代码库结构，将用户需求拆解为结构化子任务列表
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# ─────────────────────────────────────────────
# 1. 工具定义
# ─────────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """读取指定文件的内容"""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"[错误] 文件不存在: {path}"
    except Exception as e:
        return f"[错误] {str(e)}"


@tool
def list_directory(path: str = ".") -> str:
    """列出目录结构（两层深度）"""
    try:
        result = subprocess.run(
            ["find", path, "-maxdepth", "2", "-not", "-path", "*/.*",
             "-not", "-path", "*/node_modules/*", "-not", "-path", "*/__pycache__/*"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout or "[空目录]"
    except Exception as e:
        return f"[错误] {str(e)}"


@tool
def search_symbol(keyword: str, path: str = ".") -> str:
    """在代码中搜索函数/类/变量定义（类似 grep）"""
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.java",
             "--include=*.ts", "--include=*.js", keyword, path],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        # 限制返回行数，避免上下文爆炸
        limited = "\n".join(lines[:50])
        if len(lines) > 50:
            limited += f"\n... 共 {len(lines)} 条，已截断"
        return limited or f"[未找到] '{keyword}'"
    except Exception as e:
        return f"[错误] {str(e)}"


@tool
def get_references(symbol: str, path: str = ".") -> str:
    """查找某个函数/类被哪些文件调用"""
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.java",
             "--include=*.ts", "--include=*.js", symbol, path],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        limited = "\n".join(lines[:30])
        if len(lines) > 30:
            limited += f"\n... 共 {len(lines)} 条，已截断"
        return limited or f"[未找到引用] '{symbol}'"
    except Exception as e:
        return f"[错误] {str(e)}"


# ─────────────────────────────────────────────
# 2. 系统提示词
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个任务规划专家。

## 你的职责
将用户的需求拆解为可以独立执行的子任务列表。
在规划之前，先用工具了解项目结构和相关代码，再做出合理的任务拆解。

## 分析步骤
1. 用 list_directory 了解项目整体结构
2. 用 search_symbol 查找与需求相关的核心函数/类
3. 用 read_file 读取关键文件（只读最相关的，不要全读）
4. 基于理解，输出结构化任务列表

## 输出格式
分析完成后，输出如下严格 JSON（不要加任何多余文字和 markdown 代码块）：

{
  "summary": "对需求的一句话理解",
  "tech_context": {
    "tech_stack": "发现的技术栈",
    "key_modules": ["相关模块1", "相关模块2"]
  },
  "tasks": [
    {
      "id": 1,
      "title": "任务标题（动词+名词，10字以内）",
      "description": "具体做什么，30字以内",
      "input": "接收什么",
      "output": "产出什么",
      "depends_on": [],
      "estimated_complexity": "low|medium|high"
    }
  ]
}

## 约束
- 子任务数量控制在 3~7 个
- 每个任务职责单一，对应一个可交付的代码模块
- depends_on 填写依赖的任务 id，没有依赖则为空数组
- 任务按执行顺序排列
"""


# ─────────────────────────────────────────────
# 3. 构建 Agent
# ─────────────────────────────────────────────

def build_planning_agent():
    """构建规划 Agent"""
    llm = ChatTongyi(
        model="qwen3-235b-a22b",   # qwen3.6-plus 对应的 DashScope model name
        dashscope_api_key=os.environ.get("ALIYUN_KEY"),
        temperature=0,
        max_tokens=4096,
    )

    tools = [read_file, list_directory, search_symbol, get_references]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )
    return agent


# ─────────────────────────────────────────────
# 4. 解析 Agent 输出
# ─────────────────────────────────────────────

def parse_plan(agent_output: str) -> dict[str, Any]:
    """从 Agent 输出中提取 JSON 任务列表"""
    # 去掉可能存在的 markdown 代码块标记
    clean = agent_output.strip()
    clean = clean.replace("```json", "").replace("```", "").strip()

    # 找到 JSON 起始位置
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"未找到有效 JSON，原始输出：\n{agent_output}")

    return json.loads(clean[start:end])


# ─────────────────────────────────────────────
# 5. 主入口
# ─────────────────────────────────────────────

def run_planning_agent(user_input: str, project_path: str = ".") -> dict[str, Any]:
    """
    运行规划 Agent

    Args:
        user_input: 用户需求描述
        project_path: 项目根目录

    Returns:
        结构化任务列表 dict
    """
    agent = build_planning_agent()

    # 注入项目路径到用户输入
    full_input = f"项目路径：{project_path}\n\n用户需求：{user_input}"

    print(f"\n{'='*50}")
    print(f"[规划 Agent] 开始分析需求...")
    print(f"{'='*50}\n")

    result = agent.invoke({
        "messages": [HumanMessage(content=full_input)]
    })

    # 提取最后一条 AI 消息
    final_message = result["messages"][-1].content

    print(f"\n[规划 Agent] 分析完成，解析任务列表...\n")

    plan = parse_plan(final_message)

    # 打印结果
    print(f"需求理解：{plan.get('summary', '')}")
    print(f"技术栈：{plan.get('tech_context', {}).get('tech_stack', '未知')}")
    print(f"\n任务列表：")
    for task in plan.get("tasks", []):
        dep = f" (依赖: task_{task['depends_on']})" if task["depends_on"] else ""
        print(f"  [{task['id']}] {task['title']}{dep}")
        print(f"       → {task['description']}")

    return plan


# ─────────────────────────────────────────────
# 6. 示例运行
# ─────────────────────────────────────────────

if __name__ == "__main__":
    plan = run_planning_agent(
        user_input="实现用户登录功能，包括邮箱密码登录和 JWT token 管理",
        project_path="./src"
    )

    # 保存任务状态（给后续 Agent 使用）
    session = {
        "session_id": "task_001",
        "plan": plan,
        "task_status": {
            str(t["id"]): "pending"
            for t in plan.get("tasks", [])
        }
    }

    with open("task_session.json", "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] 任务会话已保存至 task_session.json")