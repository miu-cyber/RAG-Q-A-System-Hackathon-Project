import gradio as gr
import os

# Import your existing classes
from rag_spoonos_plan_multidoc import (
    RAGAgent,
    InMemoryVectorDB,
    LLM,
    create_spoonos_agent_and_bind,
    scan_documents,
    DOCS_DIR
)

# ------------------ Initialization ------------------
# 1) Scan documents
docs = scan_documents(DOCS_DIR)
if not docs:
    print("[info] No documents found. Please generate or place files into ./docs folder.")

# 2) Create vector database
vdb = InMemoryVectorDB()
vdb.add_documents(docs)

# 3) Initialize LLM
llm = LLM()

# 4) Create RAG agent
rag_agent = RAGAgent(vdb, llm)

# 5) Bind SpoonOS agent
spoon_agent = create_spoonos_agent_and_bind(rag_agent)


# ------------------ Gradio interface ------------------
def answer_question(user_question: str):
    """
    Call SpoonOS Agent to get Plan Mode results.
    Returns: plan_steps, retrieved_docs, answer_text
    """
    # Use real SpoonOS agent if available, otherwise local RAG agent
    if hasattr(spoon_agent, "handle_task"):
        result = spoon_agent.handle_task({"input": user_question})
    else:
        result = rag_agent.answer_with_plan(user_question, top_k=3)

    # Format retrieved documents for display
    docs_display = ""
    for doc in result["retrieved"]:
        docs_display += f"**{doc['id']}** (score={doc.get('score', '?'):.3f})\n{doc['snippet'][:300]}...\n\n"

    # Plan Mode steps
    plan_display = "\n".join(result["plan"])

    # Final answer
    answer_display = result["answer"]

    return plan_display, docs_display, answer_display


# ------------------ Build Gradio UI ------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🏢 Enterprise Knowledge Q&A System (RAG + SpoonOS + Gradio)")

    question_input = gr.Textbox(label="Enter your question",
                                placeholder="e.g., What is the company reimbursement policy?")
    plan_output = gr.Textbox(label="Plan Mode Steps", lines=6)
    docs_output = gr.Textbox(label="Retrieved Documents", lines=12)
    answer_output = gr.Textbox(label="Final Answer", lines=8)

    submit_btn = gr.Button("Generate Answer")

    submit_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[plan_output, docs_output, answer_output]
    )

# Launch the UI
demo.launch()
