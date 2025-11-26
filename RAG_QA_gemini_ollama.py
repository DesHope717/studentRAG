import os
import fitz
import pickle
import faiss
import numpy as np
import requests
import json
import re
from sentence_transformers import SentenceTransformer

# 全局模型和文件路径
PDF_PATH = "YJSSC.pdf"
CHUNKS_PATH = "chunks.pkl"
INDEX_PATH = "my_index.faiss"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
OLLAMA_MODEL_NAME = "gpt-oss:20b" # 确保你已经通过 ollama pull qwen2 下载了这个模型

def extract_text_from_pdf(path):
    """
    从PDF中提取文本，并按页返回。
    """
    try:
        doc = fitz.open(path)
        texts = [page.get_text("text") for page in doc]
        doc.close()
        return [text for text in texts if text.strip()]
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []

def chunk_text_recursive(text, separators=['\n\n', '\n', '。', '！', '？'], max_len=500, current_separator_index=0):
    """
    递归地将文本分块，尽量不打断语义。
    """
    if len(text) <= max_len or current_separator_index >= len(separators):
        return [text]
    
    chunks = []
    current_separator = separators[current_separator_index]
    parts = text.split(current_separator)
    
    current_chunk = ""
    for part in parts:
        if len(current_chunk) + len(part) + len(current_separator) <= max_len:
            current_chunk += part + current_separator
        else:
            chunks.append(current_chunk.strip())
            current_chunk = part + current_separator
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_len:
            final_chunks.extend(chunk_text_recursive(chunk, separators, max_len, current_separator_index + 1))
        else:
            final_chunks.append(chunk)

    return final_chunks

def build_index(pages, embed_model):
    """
    从文本页构建FAISS索引，返回分块和索引对象。
    """
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text_recursive(page))
    
    if not all_chunks:
        print("未生成任何文本分块。")
        return [], None

    embeddings = embed_model.encode(all_chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return all_chunks, index

def load_models():
    """
    加载嵌入模型并检查 Ollama 服务是否可用。
    """
    try:
        print("正在加载嵌入模型...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        print("正在检查 Ollama 服务...")
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        print("Ollama 服务已连接。")
        return embed_model
    except requests.exceptions.RequestException as e:
        print(f"无法连接到 Ollama 服务，请确保它正在运行：{e}")
        return None
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

def answer_question(query, all_chunks, index, embed_model, top_k=10, max_context_len=3000):
    """
    使用RAG和Ollama服务回答问题。
    """
    try:
        # 1. 检索
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        D, I = index.search(q_emb, top_k)
        
        relevant_chunks = [all_chunks[i] for i in I[0] if D[0][I[0].tolist().index(i)] > 0.5]
        
        if not relevant_chunks:
            return "根据提供的文档内容，未找到与问题相关的足够信息。"
        
        # 2. 上下文构建和管理
        unique_chunks = list(dict.fromkeys(relevant_chunks))
        
        current_context_len = 0
        final_chunks = []
        for chunk in unique_chunks:
            chunk_len = len(chunk)
            if current_context_len + chunk_len <= max_context_len:
                final_chunks.append(chunk)
                current_context_len += chunk_len
            else:
                break
        
        contexts = "\n\n".join(final_chunks)

        # 3. 提示词工程
        prompt = f"""
你是一个问答助手，你的唯一任务是根据提供的**上下文**来回答问题。
**严格遵守以下规则：**
1. 仅使用提供的上下文信息来回答问题。
2. 如果上下文没有足够的信息来回答，直接回复“无法根据提供的文档回答该问题。”
3. 不要添加任何个人见解或额外信息。
4. 答案要简洁、准确。
5. 只需要回答一次不要重复回答。

### 上下文:
{contexts}

### 问题:
{query}

### 答案:
"""
        # 4. 调用 Ollama API
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
        
        response = requests.post(url, data=json.dumps(payload))
        response.raise_for_status()
        
        # 解析 Ollama 的 JSON 响应
        result = response.json()
        full_response = result['response'].strip()
        
        # 5. 解码并处理输出
        answer_start_token = "### 答案:"
        if answer_start_token in full_response:
            response = full_response.split(answer_start_token)[-1].strip()
        else:
            response = full_response.strip()

        if "Human:" in response:
            response = response.split("Human:")[0].strip()
            
        return response
    except requests.exceptions.RequestException as e:
        print(f"与 Ollama API 通信失败：{e}")
        return "与本地模型通信失败，请检查 Ollama 服务。"
    except Exception as e:
        print(f"回答问题时发生错误: {e}")
        return "在回答问题时发生未知错误。"

def main():
    """主程序入口"""
    embed_model = load_models()
    if not embed_model:
        return

    print("正在处理文档...")
    if os.path.exists(CHUNKS_PATH) and os.path.exists(INDEX_PATH):
        print("发现已存在的文档索引和分块，正在加载...")
        with open(CHUNKS_PATH, "rb") as f:
            all_chunks = pickle.load(f)
        index = faiss.read_index(INDEX_PATH)
    else:
        print("未发现文档索引，正在处理PDF并构建索引...")
        pages = extract_text_from_pdf(PDF_PATH)
        if not pages:
            print("无法从PDF中提取文本，请检查文件路径或内容。")
            return
        
        all_chunks, index = build_index(pages, embed_model)
        if not all_chunks or not index:
            return
        
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)
        faiss.write_index(index, INDEX_PATH)
        print("文档索引和分块已成功保存。")
        
    print("\n准备就绪！请输入你的问题（输入 '退出' 结束）：")
    while True:
        user_query = input("\n问题: ")
        if user_query.lower() == '退出':
            print("程序已退出。")
            break
        
        print("正在思考中...")
        answer = answer_question(user_query, all_chunks, index, embed_model)
        print(f"答案: {answer}")

if __name__ == "__main__":
    main()