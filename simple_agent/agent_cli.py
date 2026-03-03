import os
import sys
from openai import OpenAI
from zero2agent import agent_loop, SYSTEM_PROMPT

def main():
    # vLLM 本地部署通常不需要 API Key，或者使用 "EMPTY"
    api_key = "EMPTY"
    
    # 初始化 OpenAI 客户端，指向本地 vLLM 服务
    client = OpenAI(
        api_key=api_key, 
        base_url="http://localhost:8000/v1"
    )
    
    # 使用 System Prompt 初始化对话历史
    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    print("----------------------------------------------------------------")
    print("AI Agent Framework ready.")
    print("Commands: 'exit' to quit, 'clear' to reset conversation context.")
    print("----------------------------------------------------------------\n")
    
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() == "exit":
            print("Bye.")
            break
            
        if user_input.lower() == "clear":
            messages.clear()
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
            print("\n(context cleared)\n")
            continue
        
        # 进入 Agent Loop (由 agent_loop 负责打印过程)
        agent_loop(user_input, messages, client)
        print() # 换行

if __name__ == "__main__":
    main()
