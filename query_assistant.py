# interactive_assistant.py (交互式 Ollama API 调用)
import os
import requests
import json
from typing import List, Dict, Optional
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ========================
# 配置区
# ========================
CHROMA_PATH = "E:/code/studentRAG/chroma_db_advanced"  # 向量库存储路径
OLLAMA_MODEL = "qwen3:14b"                            # 本地 Ollama 模型名称
OLLAMA_URL = "http://localhost:11434/api/generate"    # Ollama API URL
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"             
TOP_K = 5                                             # 召回片段数量

# ========================
# Prompt Template (提示词工程：保持不变，用于内容填充)
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
# 初始化函数 (保持不变)
# ========================
def initialize_rag_system():
    """初始化 ChromaDB 客户端和 Embedding Function"""
    print(f"🔄 初始化 RAG 系统...")
    
    # 1. 初始化 Chroma 客户端
    client = PersistentClient(path=CHROMA_PATH)
    
    # 2. 初始化 Embedding Function
    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cuda" if _is_cuda_available() else "cpu"
    )
    
    # 3. 获取 Collection
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
# Ollama API 调用函数 (核心修改点)
# ========================
def generate_response_ollama_api(prompt: str) -> str:
    """使用 requests 库调用 Ollama 的 /api/generate 接口"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,  # 设置为 False 以便一次性获取完整回复
        "options": {
            "temperature": 0.1, # 保持低温度
            "num_predict": 1024 # 限制最大输出 token 数
        }
    }
    
    try:
        # 发送 POST 请求到 Ollama API
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status() # 检查 HTTP 错误
        
        # 解析响应
        data = response.json()
        return data.get('response', 'Ollama 返回为空。')
        
    except requests.exceptions.RequestException as e:
        return f"❌ Ollama API 调用失败：请检查 Ollama 服务是否正在运行。错误: {e}"
    except json.JSONDecodeError:
        return f"❌ Ollama API 返回格式错误。"

# ========================
# RAG 查询主函数 (调整过滤逻辑)
# ========================
def rag_query(collection, query: str, handbook_type: Optional[str] = None):
    """
    执行 RAG 查询，支持元数据过滤
    :param collection: ChromaDB Collection 对象
    :param query: 用户的查询字符串
    :param handbook_type: 可选，用于过滤的手册类型 ('本科生', '研究生')
    """
    
    # 1. 设置元数据过滤 (where 子句)
    where_filter = None  # 默认设置为 None (表示不进行过滤)
    
    if handbook_type:
        # 【修改 1：使用 ChromaDB 要求的 $eq 操作符】
        where_filter = {
            "handbook_type": {
                "$eq": handbook_type
            }
        }
        print(f"🔍 使用元数据过滤: 仅搜索【{handbook_type}】手册。")

    # 2. 召回 (Retrieval)
    # print(f"🔍 正在从向量库召回 Top {TOP_K} 相关的片段...")
    try:
        results = collection.query(
            query_texts=[query],
            n_results=TOP_K,
            where=where_filter,  # 【修改 2：如果 where_filter 是 None，ChromaDB 会跳过过滤】
            include=['documents', 'metadatas']
        )
    except Exception as e:
        print(f"❌ 向量库查询失败！详细错误: {e}") 
        return "RAG 系统错误：向量库查询失败。"

    # 提取上下文 (后续代码不变)
    context_list = results.get('documents', [[]])[0]
    metadata_list = results.get('metadatas', [[]])[0]
    
    if not context_list:
        return f"抱歉，我无法从现有的手册资料中找到与问题 “{query}” 相关的信息。"

    # 格式化上下文和来源信息
    context_str = ""
    for i, (doc, meta) in enumerate(zip(context_list, metadata_list)):
        source_name = meta.get('source', '未知来源')
        handbook_type_name = meta.get('handbook_type', '未知类型')
        context_str += f"--- 片段 {i+1} (来源: {source_name}, 类型: {handbook_type_name}) ---\n"
        context_str += doc + "\n"
    
    # 3. 构造最终 Prompt
    final_prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_str,
        question=query
    )
    
    # 4. 生成 (Generation) - 调用 Ollama API
    response_content = generate_response_ollama_api(final_prompt)
    return response_content

# ========================
# 主程序入口：交互式对话
# ========================
if __name__ == "__main__":
    collection = initialize_rag_system()
    
    print("\n" + "="*50)
    print(f"🚀 智能学生助手已启动 (模型: {OLLAMA_MODEL})")
    print("提示：输入 '本科生' 或 '研究生' 切换搜索范围，输入 'quit' 退出。")
    print("="*50)

    current_filter = None
    
    while True:
        # 1. 获取用户输入
        user_input = input(f"\n[{current_filter or '通用'}] 你想问：").strip()
        
        # 2. 处理退出指令
        if user_input.lower() == 'quit':
            print("👋 助手已关闭。")
            break
        
        # 3. 处理过滤指令 (元数据切换)
        elif user_input in ["本科生", "研究生"]:
            current_filter = user_input
            print(f"✅ 搜索范围已切换为：【{current_filter}】手册。")
            continue
        
        # 4. 执行 RAG 查询
        if not user_input:
            continue

        print("🤖 正在思考...")
        
        result = rag_query(
            collection, 
            query=user_input, 
            handbook_type=current_filter
        )
        
        # 5. 输出结果
        print("\n" + "--- 智能助手回答 ---")
        print(result)
        print("----------------------")