# build_vector_db_advanced.py (方案 A)
import os
import re
from typing import List, Dict
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
# ========================
# 配置区（按需修改路径）
# ========================
PDF_DIR = "E:/code/studentRAG/data"          # 存放学生手册 PDF 的文件夹
CHROMA_PATH = "E:/code/studentRAG/chroma_db_advanced"  # 向量库存储路径
MAX_BATCH_SIZE = 5000
# ========================
# 1. 结构化文档加载与预处理
# ========================
def load_and_split_pdfs_advanced(pdf_dir: str) -> List[Document]:
    """
    使用 LangChain 加载 PDF 并进行结构化和语义分块
    """
    documents = []
    
    # 区分手册类型并设置对应的元数据
    # 你可以根据文件名规则进一步细化，例如 'undergrad' 和 'grad'
    
    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        
        filepath = os.path.join(pdf_dir, filename)
        print(f"📄 正在解析: {filename}")
        
        # 1. 使用 UnstructuredLoader 进行结构化解析
        # mode="elements" 可以将 PDF 内容分割成标题、段落、列表等元素
        loader = UnstructuredFileLoader(
            filepath, 
            mode="elements", 
            strategy="fast",# 尝试使用快速策略
            languages=["zh"]
        )
        
        # 加载文档元素
        elements = loader.load()
        
        # 2. 递归字符分割（保留段落/标题完整性）
        # 尝试使用不同的分隔符，优先按自然段落和句子分割
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", "。", "！", "？", " ", ""]
        )

        # 3. 进行分块
        chunks = recursive_splitter.split_documents(elements)
        
        # 4. 优化：添加元数据
        for i, chunk in enumerate(chunks):
            # 将源文件名作为主要元数据
            chunk.metadata['source'] = filename
            
            # 自动添加手册类型元数据 (用于 RAG 时的元数据过滤)
            if "本科生" in filename or "本科" in filename:
                chunk.metadata['handbook_type'] = "本科生"
            elif "研究生" in filename or "硕士" in filename or "博士" in filename:
                chunk.metadata['handbook_type'] = "研究生"
            else:
                 chunk.metadata['handbook_type'] = "通用"
            
            # 确保 id 是唯一的
            chunk.metadata['id'] = f"{filename}_{i}"
            
            # 重命名 content
            chunk.page_content = chunk.page_content.strip()

        print(f"  📦 切分为 {len(chunks)} 个文本块 (含元数据: handbook_type)")
        documents.extend(chunks)
    
    return documents

# ========================
# 2. 构建 Chroma 向量库
# ========================
def build_chroma_db_advanced():
    # 替换原始的 load_and_split_pdfs 函数
    docs = load_and_split_pdfs_advanced(PDF_DIR)
    
    if not docs:
        print("❌ 未找到有效文档，退出。")
        return
    
    print(f"\n🔍 开始构建向量库（使用 BGE-large-zh-v1.5）...")
    
    # 初始化 Chroma 客户端（持久化到磁盘）
    client = PersistentClient(path=CHROMA_PATH)
    
    # 使用 BGE 中文 embedding 模型
    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-large-zh-v1.5",
        device="cuda" if _is_cuda_available() else "cpu"
    )
    
    # 创建 collection
    collection = client.get_or_create_collection(
        name="student_handbook_advanced",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 提取 Chroma 需要的格式
    ids = [d.metadata['id'] for d in docs]
    documents_content = [d.page_content for d in docs]
    # 提取元数据 (注意：Chroma 的元数据要求是字典)
    metadatas = []
    
    # 定义需要排除的 LangChain/Unstructured 内部键
    EXCLUDE_KEYS = ['id', 'metadata_storage_key', 'type', 'filetype', 'languages', 'last_modified']
    
    for doc in docs:
        clean_metadata = {}
        for k, v in doc.metadata.items():
            # 1. 排除内部使用的复杂键
            if k in EXCLUDE_KEYS:
                continue
            
            # 2. 确保值是简单类型 (虽然大部分应该在你自定义的键中，但保险起见)
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean_metadata[k] = v
            # 3. 如果需要，可以进一步处理或忽略其他复杂类型
            
        metadatas.append(clean_metadata)
    
    # 添加文档（自动嵌入）
    total_chunks = len(docs)
    print(f"🚀 总共 {total_chunks} 个片段。开始分批写入 (批次大小: {MAX_BATCH_SIZE})...")
    
    # --- 【核心修改：实现分批次写入】 ---
    for i in range(0, total_chunks, MAX_BATCH_SIZE):
        batch_ids = ids[i:i + MAX_BATCH_SIZE]
        batch_documents = documents_content[i:i + MAX_BATCH_SIZE]
        batch_metadatas = metadatas[i:i + MAX_BATCH_SIZE]
        
        print(f"  ➡️ 正在处理批次 {i//MAX_BATCH_SIZE + 1}: 添加 {len(batch_ids)} 个片段...")
        
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
        except ValueError as e:
            print(f"  ❌ 批次 {i//MAX_BATCH_SIZE + 1} 添加失败: {e}")
            # 如果失败，可能需要检查 MAX_BATCH_SIZE 是否仍然过大，或者进行错误处理
            break # 停止后续处理
    
    print(f"\n✅ 向量库构建完成！共 {len(docs)} 个片段")
    print(f"📁 存储位置: {CHROMA_PATH}")
    print(f"💡 关键改进：新增 'handbook_type' 元数据，可用于 RAG 时的精确过滤。")

# ========================
# 辅助函数：检测 CUDA
# ========================
def _is_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# ========================
# 主程序入口
# ========================
if __name__ == "__main__":
    build_chroma_db_advanced()