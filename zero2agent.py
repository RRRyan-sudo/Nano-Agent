import os
import sys
import json
import subprocess
import tempfile
from openai import OpenAI

# ============================================================
# Tools 实现 — 4 个工具函数
# ============================================================

def shell_exec(command: str) -> str:
    """执行 shell 命令并返回 stdout + stderr。"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] command timed out after 30s"
    except Exception as e:
        return f"[error] {e}"

def file_read(path: str) -> str:
    """读取文件内容。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[error] {e}"

def file_write(path: str, content: str) -> str:
    """将内容写入文件（自动创建父目录）。"""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"OK — wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"[error] {e}"

def python_exec(code: str) -> str:
    """在子进程中执行 Python 代码并返回输出。"""
    try:
        # 使用 delete=False 以便在不同平台上更可靠地运行
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] execution timed out after 30s"
    except Exception as e:
        return f"[error] {e}"
    finally:
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except OSError:
            pass

# ============================================================
# Tools 注册 — name → (function, OpenAI function schema)
# ============================================================

TOOLS = {
    "shell_exec": {
        "function": shell_exec,
        "schema": {
            "type": "function",
            "function": {
                "name": "shell_exec",
                "description": "Execute a shell command and return its output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
    },
    "file_read": {
        "function": file_read,
        "schema": {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read the contents of a file at the given path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative file path.",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
    },
    "file_write": {
        "function": file_write,
        "schema": {
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "Write content to a file (creates parent directories if needed).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative file path.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
    },
    "python_exec": {
        "function": python_exec,
        "schema": {
            "type": "function",
            "function": {
                "name": "python_exec",
                "description": "Execute Python code in a subprocess and return its output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python source code to execute.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
    },
}

# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """你是一个强大的 AI 助手，可以使用以下工具来解决问题：
1. shell_exec — 执行 shell 命令
2. file_read — 读取文件内容
3. file_write — 将内容写入文件
4. python_exec — 执行 Python 代码

当前环境：Linux。
请一步步思考并解决问题。当你需要与文件系统交互、运行命令或执行代码时，请使用工具。
每次调用工具后，观察结果并决定下一步行动。
当任务完成后，请直接给出最终答案，不要再调用任何工具。
请默认使用中文进行回复。"""

# ============================================================
# Agent Loop — 核心
# ============================================================

MAX_TURNS = 20

def agent_loop(user_message: str, messages: list, client: OpenAI) -> str:
    """
    Agent Loop：while 循环驱动 LLM 推理与工具调用。
    流程：
      1. 将用户消息追加到 messages
      2. 调用 LLM
      3. 若 LLM 返回 tool_calls → 逐个执行 → 结果追加到 messages → 继续循环
      4. 若 LLM 直接返回文本（无 tool_calls）→ 退出循环，返回文本
      5. 安全上限 MAX_TURNS 轮
    """
    messages.append({"role": "user", "content": user_message})
    tool_schemas = [t["schema"] for t in TOOLS.values()]
    
    for turn in range(1, MAX_TURNS + 1):
        # --- LLM Call ---
        response = client.chat.completions.create(
            model="Qwen2.5-3B-Instruct",
            messages=messages,
            tools=tool_schemas,
        )
        
        choice = response.choices[0]
        assistant_msg = choice.message
        
        # 处理 assistant 消息中的 content 为 None 的情况（OpenAI API 要求 content 必须存在）
        msg_dict = assistant_msg.model_dump()
        if msg_dict.get("content") is None:
            msg_dict["content"] = ""
        
        # 将 assistant 消息追加到上下文
        messages.append(msg_dict)
        
        # 如果有文字内容，打印出来（中间思考或最终回复）
        if assistant_msg.content:
            print(f"\nAgent> {assistant_msg.content}")
        
        # --- 终止条件：无 tool_calls ---
        if not assistant_msg.tool_calls:
            return assistant_msg.content or ""
        
        # --- 执行每个 tool_call ---
        for tool_call in assistant_msg.tool_calls:
            name = tool_call.function.name
            raw_args = tool_call.function.arguments
            print(f"  [tool] {name}({raw_args})")
            
            # 解析参数并调用工具
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                result = f"[error] invalid JSON arguments: {raw_args}"
            else:
                tool_entry = TOOLS.get(name)
                if tool_entry is None:
                    result = f"[error] unknown tool: {name}"
                else:
                    result = tool_entry["function"](**args)
            
            # 打印工具结果的简短摘要
            summary = (result[:100] + "...") if len(result) > 100 else result
            print(f"  [result] {summary}")
            
            # 将工具结果追加到上下文
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
            
    return "[agent] reached maximum turns, stopping."
