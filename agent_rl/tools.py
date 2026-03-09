"""
工具系统 —— 从 simple_agent/zero2agent.py 移植并增强
提供工具定义、安全执行器、工具调用解析
"""
import os
import re
import sys
import json
import shutil
import tempfile
import subprocess


# ============================================================
# 工具函数实现
# ============================================================

def shell_exec(command: str, timeout: int = 5, cwd: str = None) -> str:
    """执行 shell 命令并返回 stdout + stderr。"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {timeout}s"
    except Exception as e:
        return f"[error] {e}"


def file_read(path: str, cwd: str = None) -> str:
    """读取文件内容。"""
    try:
        full_path = os.path.join(cwd, path) if cwd and not os.path.isabs(path) else path
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:2000] if len(content) > 2000 else content
    except Exception as e:
        return f"[error] {e}"


def file_write(path: str, content: str, cwd: str = None) -> str:
    """将内容写入文件（自动创建父目录）。"""
    try:
        full_path = os.path.join(cwd, path) if cwd and not os.path.isabs(path) else path
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"OK — wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"[error] {e}"


def python_exec(code: str, timeout: int = 5, cwd: str = None) -> str:
    """在子进程中执行 Python 代码并返回输出。"""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[error] execution timed out after {timeout}s"
    except Exception as e:
        return f"[error] {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ============================================================
# 工具注册 —— name → (function, OpenAI function schema)
# ============================================================

TOOLS = {
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
}

TOOL_SCHEMAS = [t["schema"] for t in TOOLS.values()]


# ============================================================
# SafeToolExecutor —— 沙箱化工具执行
# ============================================================

class SafeToolExecutor:
    """
    安全工具执行器：
    - 使用临时目录隔离文件操作
    - 超时控制（默认 5 秒）
    - 输出截断（默认 500 字符）
    """

    def __init__(self, timeout: int = 5, max_output_len: int = 500):
        self.timeout = timeout
        self.max_output_len = max_output_len
        self.workdir = tempfile.mkdtemp(prefix="agent_rl_")

    def execute(self, tool_name: str, arguments: dict) -> str:
        """执行一个工具调用并返回结果。"""
        tool_entry = TOOLS.get(tool_name)
        if tool_entry is None:
            return f"[error] unknown tool: {tool_name}"

        func = tool_entry["function"]

        # 注入 cwd 和 timeout 参数
        kwargs = dict(arguments)
        if tool_name in ("shell_exec", "python_exec"):
            kwargs["timeout"] = self.timeout
            kwargs["cwd"] = self.workdir
        elif tool_name in ("file_read", "file_write"):
            kwargs["cwd"] = self.workdir

        try:
            result = func(**kwargs)
        except Exception as e:
            result = f"[error] {e}"

        # 截断过长输出
        if len(result) > self.max_output_len:
            result = result[:self.max_output_len] + f"\n... (truncated, total {len(result)} chars)"

        return result

    def cleanup(self):
        """清理临时工作目录。"""
        try:
            shutil.rmtree(self.workdir, ignore_errors=True)
        except Exception:
            pass

    def __del__(self):
        self.cleanup()


# ============================================================
# 工具调用解析
# ============================================================

def parse_tool_calls(text: str) -> list[dict]:
    """
    从模型输出中解析工具调用。
    支持两种格式：
    1. Qwen 原生 function calling 格式（JSON tool_calls）
    2. 文本中的 ```tool_call 代码块格式
    """
    tool_calls = []

    # 格式 1: 尝试从 JSON 中提取 tool_calls
    # Qwen function calling 输出格式: {"name": "xxx", "arguments": {...}}
    # 匹配 {"name": "...", "arguments": ...} 模式
    json_pattern = r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
    for match in re.finditer(json_pattern, text):
        try:
            name = match.group(1)
            args = json.loads(match.group(2))
            if name in TOOLS:
                tool_calls.append({"name": name, "arguments": args})
        except (json.JSONDecodeError, IndexError):
            continue

    if tool_calls:
        return tool_calls

    # 格式 2: 尝试从整个文本中解析完整 JSON
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            if data["name"] in TOOLS:
                tool_calls.append({"name": data["name"], "arguments": data["arguments"]})
                return tool_calls
    except json.JSONDecodeError:
        pass

    # 格式 3: 匹配 <tool_call> 标签格式
    tc_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    for match in re.finditer(tc_pattern, text, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                if data["name"] in TOOLS:
                    tool_calls.append({"name": data["name"], "arguments": data["arguments"]})
        except json.JSONDecodeError:
            continue

    return tool_calls


def execute_tool_calls(
    tool_calls: list[dict],
    executor: SafeToolExecutor | None = None,
) -> list[dict]:
    """
    执行一组工具调用并返回结果列表。
    每个结果: {"name": str, "result": str}
    """
    if executor is None:
        executor = SafeToolExecutor()
        should_cleanup = True
    else:
        should_cleanup = False

    results = []
    for tc in tool_calls:
        result = executor.execute(tc["name"], tc["arguments"])
        results.append({"name": tc["name"], "result": result})

    if should_cleanup:
        executor.cleanup()

    return results
