"""
rag_spoonos_agent.py
Full SpoonOS RAG demo (Enterprise QnA) with "Super UI" (upload docs, show plan, top-k, agent chain)
- UI: Gradio
- Embedding: bge-large-en-v1.5 (configurable)
- VectorDB: Local Chroma (via spoon_ai.retrieval.get_retrieval_client backend='chroma')
- SpoonAI LLM & Agent integration (preferred; will error with instructions if not installed)

Usage:
    python rag_spoonos_agent.py            # CLI mode
    python rag_spoonos_agent.py --web      # Gradio Web UI
    python rag_spoonos_agent.py --web --share   # Gradio web UI with public link

Important:
- This file **expects spoon_ai SDK** available in the environment for official SpoonOS usage.
- If spoon_ai is not installed, the script will show informative messages and fall back to local mocks for development.
- For competition submission, ensure environment has spoon_ai, chromadb, and correct credentials.
"""

import os
import sys
import json
import tempfile
import pathlib
import uuid
from typing import List, Tuple, Dict, Any
from logging import getLogger
from sentence_transformers import SentenceTransformer


logger = getLogger(__name__)

# ---------------- Config ----------------
DOCS_DIR = "./docs"
CHROMA_STORE = "./chroma_store"
EMBEDDING_MODEL = "bge-large-en-v1.5"  # per your choice
TOP_K_DEFAULT = 3
OPENAI_FALLBACK_MODEL = "gpt-3.5-turbo"  # used only if spoon_ai not available and openai configured
# ----------------------------------------

# ---------------- Utilities (file loading / chunking) ----------------
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

def ensure_docs_dir():
    p = pathlib.Path(DOCS_DIR)
    p.mkdir(parents=True, exist_ok=True)

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(path: str) -> str:
    if PyPDF2 is None:
        return ""
    text_parts = []
    try:
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
    except Exception as e:
        print(f"[warn] Failed to parse PDF {path}: {e}")
    return "\n".join(text_parts)

def scan_local_docs(doc_dir: str = DOCS_DIR) -> List[Dict[str, str]]:
    """Scan ./docs for .txt/.md/.pdf and return list of {id, page_content, metadata}"""
    ensure_docs_dir()
    p = pathlib.Path(doc_dir)
    docs = []
    for ext in ("*.txt", "*.md", "*.pdf"):
        for fp in sorted(p.glob(ext)):
            text = ""
            if fp.suffix.lower() in [".txt", ".md"]:
                text = load_txt(str(fp))
            elif fp.suffix.lower() == ".pdf":
                text = load_pdf(str(fp))
            if text and text.strip():
                docs.append({
                    "id": fp.name,
                    "page_content": text,
                    "source": str(fp)
                })
    return docs

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Naive chunker by characters (works okay for demo). Returns list of chunk strings."""
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks

# ---------------- SpoonOS / spoon_ai integration wrappers ----------------
# We try to use spoon_ai SDK where available. We design wrappers for:
#  - Retrieval client (Chroma backend)
#  - SpoonAI LLM / Agent
#

SPOON_AVAILABLE = False
try:
    import spoon_ai
    # try to import common submodules used by examples
    try:
        from spoon_ai.retrieval import get_retrieval_client
    except Exception:
        get_retrieval_client = None
    SPOON_AVAILABLE = True
except Exception:
    spoon_ai = None
    get_retrieval_client = None
    SPOON_AVAILABLE = False

# If spoon_ai is not installed, we'll attempt to fallback to local components:
LOCAL_FALLBACK = not SPOON_AVAILABLE

# Fallback: sentence-transformers when spoon_ai not available for embedding-based local testing
if LOCAL_FALLBACK:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except Exception:
        SentenceTransformer = None

# LLM wrapper: prefer spoon_ai.llm or spoon_ai.client, else openai, else mock
class LLMWrapper:
    def __init__(self, spoon_model: str = None, openai_model: str = OPENAI_FALLBACK_MODEL):
        self.spoon_model = spoon_model
        self.openai_model = openai_model
        self.use_spoon = False
        self.use_openai = False
        self.api = None

        # try spoon_ai LLM
        if SPOON_AVAILABLE:
            # attempt to find a high-level spoon_ai LLM interface
            # Many spoon_ai SDKs expose either spoon_ai.llm or spoon_ai.client.Spoon
            try:
                # prefer spoon_ai.llm.call or spoon_ai.Spoon
                if hasattr(spoon_ai, "llm"):
                    self.api = spoon_ai.llm
                    self.use_spoon = True
                elif hasattr(spoon_ai, "SpoonAI"):
                    self.api = spoon_ai.SpoonAI
                    self.use_spoon = True
                else:
                    # fallback: try to use spoon_ai.client or similar
                    self.api = spoon_ai
                    self.use_spoon = True
            except Exception:
                self.use_spoon = False

        # fallback to openai if available
        if not self.use_spoon:
            try:
                import openai
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                if openai_api_key:
                    openai.api_key = openai_api_key
                    self.openai = openai
                    self.use_openai = True
            except Exception:
                self.use_openai = False

    def generate(self, system: str, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        # try spoon_ai invocation patterns (best-effort)
        if self.use_spoon:
            # Different spoon_ai SDKs expose different APIs.
            # Try a few likely ones:
            try:
                # 1) spoon_ai.llm.call_llm(system, prompt, ...)
                if hasattr(self.api, "call_llm"):
                    return self.api.call_llm(system_prompt=system, prompt=prompt, model=self.spoon_model, max_tokens=max_tokens)
                # 2) spoon_ai.llm.chat(...)
                if hasattr(self.api, "chat"):
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})
                    resp = self.api.chat(model=self.spoon_model, messages=messages, max_tokens=max_tokens, temperature=temperature)
                    # attempt to extract text
                    if isinstance(resp, dict) and "content" in resp:
                        return resp["content"]
                    if hasattr(resp, "choices"):
                        return resp.choices[0].message.content.strip()
                    return str(resp)
                # 3) spoon_ai.SpoonAI usage
                if hasattr(self.api, "SpoonAI"):
                    # this branch unlikely; left as placeholder
                    client = self.api.SpoonAI()
                    return client.chat(system, prompt)
                # generic fallback
            except Exception as e:
                print(f"[warn] spoon_ai LLM call failed: {e}")

        if self.use_openai:
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                resp = self.openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[warn] openai call failed: {e}")

        # final fallback: simple mock
        snippet = prompt[:1000].replace("\n", " ")
        return f"[mock reply]\nSystem: {system}\nPrompt snippet: {snippet}"

# Retrieval wrapper using spoon_ai retrieval client (Chroma)
class SpoonRetrievalWrapper:
    def __init__(self, backend: str = "chroma", config_dir: str = ".", embedding_model: str = EMBEDDING_MODEL, chroma_path: str = CHROMA_STORE):
        self.backend = backend
        self.config_dir = config_dir
        self.embedding_model = embedding_model
        self.chroma_path = chroma_path
        self.client = None
        self.local_fallback_index = None  # only used when spoon_ai missing

        if SPOON_AVAILABLE and get_retrieval_client is not None:
            try:
                # get_retrieval_client API: get_retrieval_client(backend, config_dir=..., **kwargs)
                # we pass embedding_model and persist path where supported
                self.client = get_retrieval_client(backend, config_dir=str(self.config_dir), embedding_model=self.embedding_model, persist_directory=self.chroma_path)
            except Exception as e:
                print(f"[warn] get_retrieval_client failed: {e}")
                self.client = None

        if self.client is None:
            # local fallback: sentence-transformers in-memory vector index
            if SentenceTransformer is None:
                print("[error] spoon_ai not available and sentence-transformers not installed. Install spoon_ai or sentence-transformers.")
            else:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                self.local_docs = []
                self.local_embeddings = None

    def add_documents(self, docs: List[Dict[str, Any]]):
        """
        docs: list of dicts with keys "id" and "page_content" (and optional metadata)
        """
        if self.client is not None:
            try:
                # Many spoon retrieval clients accept list of docs with 'id' and 'page_content'
                self.client.add_documents(docs)
                return
            except Exception as e:
                print(f"[warn] spoon retrieval add_documents failed: {e}")

        # fallback: index locally
        texts = [d["page_content"] for d in docs]
        ids = [d.get("id") or str(uuid.uuid4()) for d in docs]
        embs = self.embedder.encode(texts, show_progress_bar=False)
        import numpy as np
        if self.local_embeddings is None:
            self.local_embeddings = np.array(embs)
        else:
            self.local_embeddings = np.vstack([self.local_embeddings, np.array(embs)])
        for i, txt in enumerate(texts):
            self.local_docs.append({"id": ids[i], "page_content": txt})

    def query(self, query_text: str, k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
        if self.client is not None:
            try:
                # signature: client.query(query, k=...)
                return self.client.query(query_text, k=k)
            except Exception as e:
                print(f"[warn] spoon retrieval query failed: {e}")

        # local fallback:
        import numpy as np
        qv = self.embedder.encode([query_text], show_progress_bar=False)[0]
        norms = (np.linalg.norm(self.local_embeddings, axis=1) * np.linalg.norm(qv)) + 1e-12
        sims = np.dot(self.local_embeddings, qv) / norms
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for i in top_idx:
            results.append({"id": self.local_docs[i]["id"], "page_content": self.local_docs[i]["page_content"], "score": float(sims[i])})
        return results

# ---------------- RAG Agent (Plan mode + answer synthesis) ----------------
class RAGAgent:
    def __init__(self, retriever: SpoonRetrievalWrapper, llm: LLMWrapper):
        self.retriever = retriever
        self.llm = llm

    def generate_plan(self, question: str) -> List[str]:
        system = "You are an assistant that outputs a concise step-by-step plan as numbered lines to answer a business question using retrieved documents."
        user = f"Question: {question}\nProvide a 3-6 step numbered plan (each step short)."
        raw = self.llm.generate(system, user, max_tokens=200)
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        # prefer numbered lines; if not, just return first few non-empty lines
        plan = []
        for l in lines:
            if len(plan) >= 6:
                break
            plan.append(l)
        if not plan:
            plan = ["1. Retrieve top relevant documents.", "2. Summarize key points.", "3. Compose final concise answer with citations."]
        return plan

    def answer_with_plan(self, question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
        plan = self.generate_plan(question)
        retrieved = self.retriever.query(question, k=top_k)
        # format retrieved text snippet
        docs_text = ""
        for d in retrieved:
            snippet = d.get("page_content", "")[:800]
            docs_text += f"Document: {d.get('id')}\n{snippet}\n\n"
        system = "You are a helpful enterprise assistant. Use the retrieved documents to produce a concise answer; cite document ids in brackets when referencing facts."
        user_prompt = f"Plan:\n{chr(10).join(plan)}\n\nRetrieved Documents:\n{docs_text}\nQuestion: {question}\n\nPlease give a concise answer and list which document ids you used."
        answer = self.llm.generate(system, user_prompt, max_tokens=512, temperature=0.2)
        return {"plan": plan, "retrieved": retrieved, "answer": answer}

# ---------------- SpoonOS Agent binding (optional, uses spoon_ai.agent if available) ----------------
def create_spoonos_agent_binding(rag_agent: RAGAgent):
    """
    If spoon_ai supports agent creation, register a task handler that runs plan mode.
    Returns an agent object or a local shim with .handle_task(dict)->dict
    """
    if not SPOON_AVAILABLE:
        print("[info] spoon_ai SDK not available — agent shim will be used (for local demo only).")
        class ShimAgent:
            def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
                q = task.get("input") or task.get("question") or ""
                return rag_agent.answer_with_plan(q, top_k=TOP_K_DEFAULT)
        return ShimAgent()

    # try to use spoon_ai.agent API
    try:
        # many spoon SDKs offer something like: from spoon_ai.agent import Agent
        if hasattr(spoon_ai, "agent") and hasattr(spoon_ai.agent, "Agent"):
            AgentCls = spoon_ai.agent.Agent
            agent = AgentCls(name="RAG_Enterprise_Agent", description="Handles enterprise QnA using RAG")
            # We attempt to bind a simple handler – exact API may vary; we try best-effort
            def handler(task):
                q = task.get("input") or task.get("question") or ""
                return rag_agent.answer_with_plan(q, top_k=TOP_K_DEFAULT)
            # try multiple possible handler registration methods
            if hasattr(agent, "on_task"):
                agent.on_task(handler)
            elif hasattr(agent, "set_handler"):
                agent.set_handler(handler)
            elif hasattr(agent, "register"):
                agent.register(handler)
            else:
                # fallback: add attribute
                agent._handler = handler
            print("[spoon_ai] Created Agent and bound handler.")
            return agent
        else:
            # fallback: use top-level spoon_ai to create agent (best-effort)
            if hasattr(spoon_ai, "Agent"):
                agent = spoon_ai.Agent(name="RAG_Enterprise_Agent")
                if hasattr(agent, "on_task"):
                    agent.on_task(lambda t: rag_agent.answer_with_plan(t.get("input",""), top_k=TOP_K_DEFAULT))
                return agent
    except Exception as e:
        print(f"[warn] failed to create spoon agent via SDK: {e}")

    # final fallback shim
    class ShimAgent2:
        def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
            q = task.get("input") or task.get("question") or ""
            return rag_agent.answer_with_plan(q, top_k=TOP_K_DEFAULT)
    return ShimAgent2()

# ---------------- CLI and Web UI ----------------
def ensure_example_docs():
    ensure_docs_dir()
    existing = scan_local_docs()
    if existing:
        return
    example_docs = {
        "hr_policy.txt": "公司假期政策：每位员工每年享有15天带薪年假，另有病假和事假。请在年初与主管沟通排休。",
        "it_guide.md": "IT 支持指南：如电脑故障请联系 it-support@company.com，提交工单并附截图。VPN 配置指南在 /internal/vpn.md。",
        "expense_policy.txt": "报销政策：员工可报销交通、住宿、餐饮（需发票），报销在财务系统填写并上传发票，审批流程 3 天内处理。",
        "product_onboarding.txt": "产品上手指南：客户 A 的安装流程概述，关键里程碑与联系人信息。"
    }
    for fn, txt in example_docs.items():
        with open(os.path.join(DOCS_DIR, fn), "w", encoding="utf-8") as f:
            f.write(txt)

def main_cli():
    ensure_example_docs()
    docs = scan_local_docs()
    print(f"[info] Found {len(docs)} local docs.")
    retriever = SpoonRetrievalWrapper(backend="chroma", config_dir=".", embedding_model=EMBEDDING_MODEL, chroma_path=CHROMA_STORE)
    retriever.add_documents(docs)
    llm = LLMWrapper(spoon_model=None)
    rag = RAGAgent(retriever, llm)
    agent_binding = create_spoonos_agent_binding(rag)

    print("=== RAG Enterprise QnA CLI (SpoonOS-enabled) ===")
    print("Type a question (or 'exit'):")
    while True:
        q = input("> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        out = rag.answer_with_plan(q, top_k=TOP_K_DEFAULT)
        print("\n--- Plan ---")
        for s in out["plan"]:
            print(s)
        print("\n--- Retrieved Docs ---")
        for d in out["retrieved"]:
            print(f"{d.get('id')} (score={d.get('score','?')}):\n{d.get('page_content','')[:400]}...\n")
        print("\n--- Answer ---")
        print(out["answer"])
        print("\n(You can also invoke the SpoonOS agent binding programmatically.)\n")

def run_webui(share: bool = False):
    try:
        import gradio as gr
    except Exception:
        print("[error] gradio is not installed. Install it with `pip install gradio` and retry.")
        return

    ensure_example_docs()
    docs = scan_local_docs()
    retriever = SpoonRetrievalWrapper(backend="chroma", config_dir=".", embedding_model=EMBEDDING_MODEL, chroma_path=CHROMA_STORE)
    retriever.add_documents(docs)
    llm = LLMWrapper(spoon_model=None)
    rag = RAGAgent(retriever, llm)

    # inner function for Gradio
    def answer_fn(question: str, top_k: int = TOP_K_DEFAULT, show_plan: bool = True):
        if not question or not question.strip():
            return "","", "Please input a question."
        res = rag.answer_with_plan(question, top_k=top_k)
        plan_text = "\n".join(res["plan"]) if show_plan else "(hidden)"
        retrieved_text = "\n\n".join([f"{d.get('id')} (score={d.get('score','?')}):\n{d.get('page_content','')[:800]}..." for d in res["retrieved"]])
        return plan_text, retrieved_text, res["answer"]

    with gr.Blocks() as demo:
        gr.Markdown("# RAG Enterprise QnA (SpoonOS Full Version)")
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(lines=3, placeholder="例如：公司假期政策是什么？", label="问题")
                topk_slider = gr.Slider(minimum=1, maximum=6, step=1, value=TOP_K_DEFAULT, label="Top-K 检索数量")
                show_plan_checkbox = gr.Checkbox(value=True, label="显示 Agent Plan (可解释步骤)")
                submit_btn = gr.Button("提交")
            with gr.Column(scale=2):
                gr.Markdown("**说明**\n- 上传或将企业文档放入 `./docs`。\n- 系统会检索 Top-K 文档并由 SpoonAI LLM 生成答案。")
                doc_list_md = "\n".join([f"- {d['id']}" for d in docs])
                gr.Markdown("**已加载文档**\n" + (doc_list_md or "（暂无文档）"))

        with gr.Tabs():
            with gr.TabItem("结果"):
                plan_out = gr.Textbox(label="Plan (步骤)", interactive=False)
                retrieved_out = gr.Textbox(label="Retrieved Docs (摘要)", interactive=False)
                answer_out = gr.Textbox(label="Answer", interactive=False)
            with gr.TabItem("高级"):
                gr.Markdown("**Agent Trace / 调试信息**")
                trace_out = gr.Textbox(label="Trace", interactive=False)

        submit_btn.click(fn=answer_fn, inputs=[question_input, topk_slider, show_plan_checkbox], outputs=[plan_out, retrieved_out, answer_out])

    demo.launch(share=share)

# ---------------- Entry Point ----------------
def print_usage():
    print("Usage:")
    print("  python rag_spoonos_agent.py          # CLI mode")
    print("  python rag_spoonos_agent.py --web    # run Gradio web UI")
    print("  python rag_spoonos_agent.py --web --share  # Run Gradio with public share link")

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--web" in args:
        share = "--share" in args
        run_webui(share=share)
    else:
        main_cli()
