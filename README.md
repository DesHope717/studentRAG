\# Student RAG Assistant



A retrieval-augmented generation (RAG) system for answering student queries using university handbooks.



\## Features

\- Support multiple embedding backends (Ollama, OPENAI API)

\- Chroma vector database

\- Re-query mechanism for ambiguous questions



\## Setup

```bash

pip install -r requirements.txt

python build\_vector\_db\_advanced.py  # builds local vector DB from PDFs in ./data

python RAG\_QA.py

