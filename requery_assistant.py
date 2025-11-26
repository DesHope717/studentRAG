# interactive_assistant.py (集成 Query Rewriting)
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
# Prompt Template
# ========================

# 1. 召回增强：查询重写提示词
QUERY_REWRITE_PROMPT = """
你是一个专业的RAG系统查询重写器。你的任务是将用户口语化、模糊或不完整的提问，改写成一个或多个(不超过3个)，更正式、更专业的、更可能命中大学学生手册中标准条款或标题的搜索查询。

仅输出改写后的查询，无需任何解释或前缀。

【原始问题】：{query}
【改写后的专业查询】：
"""

# 2. 最终生成：RAG 提示词
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
# 辅助函数
# ========================
def _is_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

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

def generate_response_ollama_api(prompt: str) -> str:
    """使用 requests 库调用 Ollama 的 /api/generate 接口"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1, 
            "num_predict": 1024 
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300) # 增加 timeout
        response.raise_for_status() 
        data = response.json()
        return data.get('response', 'Ollama 返回为空。')
        
    except requests.exceptions.RequestException as e:
        return f"❌ Ollama API 调用失败：请检查 Ollama 服务是否正在运行。错误: {e}"
    except json.JSONDecodeError:
        return f"❌ Ollama API 返回格式错误。"

# ========================
# 【新增】查询转换函数
# ========================
def rewrite_query(query: str) -> str:
    """将用户查询转换为更专业的搜索词"""
    rewrite_prompt = QUERY_REWRITE_PROMPT.format(query=query)
    
    # 使用 Ollama API 进行改写
    # 注意：我们使用 temperature=0.1 保持改写结果的稳定性
    rewritten_query = generate_response_ollama_api(rewrite_prompt)
    
    # 清理并返回第一个改写结果（如果模型输出了多个，只取第一行）
    rewritten_query = rewritten_query.strip().split('\n')[0].replace('【改写后的专业查询】：', '').strip()
    
    # 避免空查询，如果改写失败，仍使用原查询
    if not rewritten_query or rewritten_query.lower() == 'ollama 返回为空。':
        return query
    
    return rewritten_query

# ========================
# RAG 查询主函数 (调整：加入查询转换)
# ========================
def rag_query(collection, query: str, handbook_type: Optional[str] = None):
    """
    执行 RAG 查询，先进行查询转换，再召回。
    """
    
    # 0. 【新步骤】执行查询转换
    rewritten_query = rewrite_query(query)
    
    if rewritten_query != query:
        print(f"  ✨ 查询已改写：'{query}' -> '{rewritten_query}'")
    else:
        print(f"  ➡️ 使用原始查询：'{query}'")
    
    # 1. 设置元数据过滤 (where 子句)
    where_filter = None 
    if handbook_type:
        where_filter = {"handbook_type": {"$eq": handbook_type}}
        print(f"🔍 使用元数据过滤: 仅搜索【{handbook_type}】手册。")

    # 2. 召回 (Retrieval) - **使用改写后的查询**
    try:
        results = collection.query(
            query_texts=[rewritten_query], # <--- 使用改写后的查询
            n_results=TOP_K,
            where=where_filter,
            include=['documents', 'metadatas']
        )
    except Exception as e:
        print(f"❌ 向量库查询失败！详细错误: {e}") 
        return "RAG 系统错误：向量库查询失败。"

    # 提取上下文 (后续代码保持不变)
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
    
    # 3. 构造最终 Prompt - **使用原始查询**
    final_prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_str,
        question=query # <--- 最终回答时，依然使用原始查询，确保答案符合用户语境
    )
    
    # 4. 生成 (Generation)
    response_content = generate_response_ollama_api(final_prompt)
    return response_content


# ========================
# 主程序入口：交互式对话 (保持不变)
# ========================
if __name__ == "__main__":
    collection = initialize_rag_system()
    
    print("\n" + "="*50)
    print(f"🚀 智能学生助手已启动 (模型: {OLLAMA_MODEL})")
    print("提示：输入 '本科生' 或 '研究生' 切换搜索范围，输入 'quit' 退出。")
    print("【已启用查询转换 (Query Rewriting) 增强召回】")
    print("="*50)

    current_filter = None
    
    while True:
        user_input = input(f"\n[{current_filter or '通用'}] 你想问：").strip()
        
        if user_input.lower() == 'quit':
            print("👋 助手已关闭。")
            break
        
        elif user_input in ["本科生", "研究生"]:
            current_filter = user_input
            print(f"✅ 搜索范围已切换为：【{current_filter}】手册。")
            continue
        
        if not user_input:
            continue

        print("🤖 正在思考...")
        
        result = rag_query(
            collection, 
            query=user_input, 
            handbook_type=current_filter
        )
        
        print("\n" + "--- 智能助手回答 ---")
        print(result)
        print("----------------------")