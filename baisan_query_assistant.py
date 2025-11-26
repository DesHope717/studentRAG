# interactive_assistant.py - 使用 EdgeFn 云端大模型的 RAG 助手
import os
import requests
import json
from typing import List, Optional
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ========================
# 配置区
# ========================
CHROMA_PATH = "E:/code/studentRAG/chroma_db_advanced"  # 向量库存储路径
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
TOP_K = 5  # 召回片段数量

# --- 云端大模型配置 ---
CLOUD_API_BASE = "https://api.edgefn.net/v1"
CLOUD_MODEL_NAME = "Qwen3-Next-80B-A3B-Instruct"
CLOUD_API_KEY = os.getenv("BAISAN_API")  # 从系统环境变量 BAISAN_API 读取

if not CLOUD_API_KEY:
    raise EnvironmentError(
        "❌ 环境变量 'BAISAN_API' 未设置！\n"
        "请在运行前设置 API 密钥，例如（PowerShell）：\n"
        "$env:BAISAN_API='your_actual_api_key_here'\n"
        "或（CMD）：\n"
        "set BAISAN_API=your_actual_api_key_here"
    )

# ========================
# Prompt Template
# ========================
RAG_PROMPT_TEMPLATE = """
你是一位专业、严谨的大学学生智能助手。你的任务是根据提供的**上下文资料**，准确、简洁地回答学生的提问。

请遵守以下严格的约束和步骤：
1.  **严格基于上下文：** 你的回答必须且仅基于【上下文资料】中提供的信息。
2.  **提取并引用：** 在回答中，用中文分点列出关键信息。
3.  **注明来源：** 在回答的最后，务必以“资料来源：[文件名]”的格式，注明你使用的所有上下文资料的文件名（来自上下文中的 'source'）。
4.  **无法回答时：** 如果上下文资料无法回答问题，请礼貌地回答：“抱歉，我无法从现有的学生手册资料中找到相关信息。”

---
【上下文资料】
{context}
---
【学生问题】
{question}
"""

# ========================
# 初始化函数
# ========================
def initialize_rag_system():
    """初始化 ChromaDB 客户端和 Embedding Function"""
    print(f"🔄 初始化 RAG 系统...")
    
    client = PersistentClient(path=CHROMA_PATH)
    
    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cuda" if _is_cuda_available() else "cpu"
    )
    
    collection = client.get_collection(
        name="student_handbook_advanced",
        embedding_function=embedding_func
    )
    
    print(f"✅ RAG 系统初始化完成。当前向量库片段数: {collection.count()}")
    return collection

def _is_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# ========================
# 云端 API 调用函数（核心）
# ========================
def generate_response_cloud_api(prompt: str) -> str:
    """调用 EdgeFn 云平台的 OpenAI 兼容 Chat Completions API"""
    url = f"{CLOUD_API_BASE}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {CLOUD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": CLOUD_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个严谨的大学学生事务助手，请严格根据提供的资料回答问题。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return content if content else "⚠️ 云端模型返回了空内容。"
        
    except requests.exceptions.Timeout:
        return "❌ 请求超时：云端 API 响应过慢，请稍后再试。"
    except requests.exceptions.HTTPError as e:
        try:
            error_msg = response.json().get("error", {}).get("message", str(e))
        except:
            error_msg = response.text[:200]
        return f"❌ HTTP 错误 {response.status_code}: {error_msg}"
    except requests.exceptions.RequestException as e:
        return f"❌ 网络请求失败：{e}"
    except json.JSONDecodeError:
        return f"❌ API 返回非 JSON 格式：{response.text[:200]}"

# ========================
# RAG 查询主函数
# ========================
def rag_query(collection, query: str, handbook_type: Optional[str] = None):
    """执行 RAG 查询，支持元数据过滤"""
    where_filter = None
    if handbook_type:
        where_filter = {"handbook_type": {"$eq": handbook_type}}
        print(f"🔍 使用元数据过滤: 仅搜索【{handbook_type}】手册。")

    try:
        results = collection.query(
            query_texts=[query],
            n_results=TOP_K,
            where=where_filter,
            include=['documents', 'metadatas']
        )
    except Exception as e:
        return f"❌ 向量库查询失败：{e}"

    context_list = results.get('documents', [[]])[0]
    metadata_list = results.get('metadatas', [[]])[0]
    
    if not context_list:
        return f"抱歉，我无法从现有的手册资料中找到与问题 “{query}” 相关的信息。"

    context_str = ""
    for i, (doc, meta) in enumerate(zip(context_list, metadata_list)):
        source_name = meta.get('source', '未知来源')
        context_str += f"--- 片段 {i+1} (来源: {source_name}) ---\n{doc}\n"

    final_prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, question=query)
    return generate_response_cloud_api(final_prompt)

# ========================
# 主程序入口
# ========================
if __name__ == "__main__":
    collection = initialize_rag_system()
    
    print("\n" + "="*50)
    print(f"🚀 智能学生助手已启动 (云端模型: {CLOUD_MODEL_NAME})")
    print("提示：输入 '本科生' 或 '研究生' 切换搜索范围，输入 'quit' 退出。")
    print("="*50)

    current_filter = None
    
    while True:
        user_input = input(f"\n[{current_filter or '通用'}] 你想问：").strip()
        
        if user_input.lower() == 'quit':
            print("👋 助手已关闭。")
            break
        
        if user_input in ["本科生", "研究生"]:
            current_filter = user_input
            print(f"✅ 搜索范围已切换为：【{current_filter}】手册。")
            continue
        
        if not user_input:
            continue

        print("🤖 正在思考...")
        result = rag_query(collection, user_input, current_filter)
        
        print("\n" + "--- 智能助手回答 ---")
        print(result)
        print("----------------------")