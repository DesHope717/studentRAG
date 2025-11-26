import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    texts = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")
        if text.strip():
            texts.append(text)
    return texts

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_index(pages):
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text(page))

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_model.encode(all_chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return all_chunks, embed_model, index

def load_model():
    model_name = "./Qwen2.5-7B-Instruct-GPTQ-Int4"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

def answer_question(query, all_chunks, index, embed_model, pipe, top_k=3):
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    contexts = "\n\n".join([all_chunks[i] for i in I[0]])

    prompt = f"""
You are a helpful assistant. Use ONLY the following book content to answer.

Context:
{contexts}

Question: {query}
Answer:
"""

    out = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0)[0]["generated_text"]
    return out

if __name__ == "__main__":
    # 1. 提取 PDF
    pages = extract_text_from_pdf("PKRL.pdf")

    # 2. 构建索引
    all_chunks, embed_model, index = build_index(pages)

    # 3. 加载模型
    pipe = load_model()

    # 4. 问答测试
    print(answer_question("这本书的讲强化是什么？", all_chunks, index, embed_model, pipe))
