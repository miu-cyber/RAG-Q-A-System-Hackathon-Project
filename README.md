Logic：
User → SpoonOS Agent → (Query transform) → VectorDB 
                                   ↓
                                SpoonAI LLM
                                   ↓
                              Final Answer
```
                              
📁 Project File Structure

├── README.md                     # Project overview, setup, usage
├── rag_spoonos_gradio.py         # Main Gradio demo (UI + RAG pipeline)
├── rag_spoonos_plan_multidoc.py  # Multi-document RAG pipeline using SpoonOS agents
├── demo.py                       # Minimal CLI demo for debugging
│
├── web.jsx                       # Web frontend (React/Next.js/Vercel friendly)
├── web.png                       # UI screenshot for README/demo
│
├── documents/                    # Knowledge base documents for retrieval
│   ├── 2024-Annual-Report-Target-Corporation.pdf
│   ├── 2024-pepsico-annual-report-01.pdf
│   ├── company_policy.md
│   ├── hr_faq.txt
│   ├── it_guide.md
│   ├── product_info.txt
│   └── tech_doc.md
│
├── embeddings/                   # Cached embeddings for performance
│   └── *.pkl
│
├── vectorstore/                  # Local Chroma / Milvus index data
│   └── index/
│
├── assets/                       # Images / static resources
│   └── web.png
│
└── utils/                        # Utility scripts
    └── (optional helpers)
```
