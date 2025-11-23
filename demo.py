# RAG Q&A Agent Demo with SpoonOS Integration
# Minimal single-file version including vector retrieval + LLM simulation + SpoonOS Agent

from sentence_transformers import SentenceTransformer
import numpy as np
import spoonos

# ----------------------------
# Simple Vector Database (Minimal Implementation)
# ----------------------------
class SimpleVectorDB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = []
        self.embeddings = []

    def add_document(self, text):
        # Store document and its embedding
        self.docs.append(text)
        self.embeddings.append(self.model.encode(text))

    def query(self, query_text, top_k=1):
        # Compute query embedding and return the most similar document
        q_vec = self.model.encode(query_text)
        scores = [np.dot(q_vec, e) / (np.linalg.norm(q_vec) * np.linalg.norm(e)) for e in self.embeddings]
        idx = np.argmax(scores)
        return self.docs[idx]

# ----------------------------
# Simulated LLM
# ----------------------------
class SimpleLLM:
    def generate(self, prompt):
        # Simulated generation response
        return f"【Simulated Answer】Based on document content: {prompt}"

# ----------------------------
# RAG (Retrieval-Augmented Generation) Agent
# ----------------------------
class RAGAgent:
    def __init__(self, llm, vector_db: SimpleVectorDB):
        self.llm = llm
        self.vector_db = vector_db

    def answer_question(self, question: str):
        # Retrieve the most relevant document and build a prompt
        docs = self.vector_db.query(question)
        prompt = f"Answer the question based on the following content: {docs}\nQuestion: {question}\nAnswer:"
        return self.llm.generate(prompt)

# ----------------------------
# Initialization
# ----------------------------
vector_db = SimpleVectorDB()

# Add sample documents
vector_db.add_document("The latest AI research directions include multimodal learning, reinforcement learning optimization, generative models, and explainable AI.")
vector_db.add_document("RAG systems combine vector retrieval and LLM outputs to improve answer accuracy and interpretability.")

llm = SimpleLLM()
rag_agent = RAGAgent(llm, vector_db)

# ----------------------------
# SpoonOS Integration
# ----------------------------
spoonos.api_key = "YOUR_API_KEY"

# Create Agent
my_agent = spoonos.Agent.create(
    name="RAG_QnA_Agent",
    type="generative",
    model="llm-model-name"
)

# Bind task handler
def handle_question(task):
    question = task.get("input")
    answer = rag_agent.answer_question(question)
    return {"output": answer}

my_agent.on_task(handle_question)

# ----------------------------
# CLI Test
# ----------------------------
if __name__ == "__main__":
    print("=== RAG Q&A Agent Demo ===")
    while True:
        question = input("Enter your question: ")
        answer = rag_agent.answer_question(question)
        print(answer)
