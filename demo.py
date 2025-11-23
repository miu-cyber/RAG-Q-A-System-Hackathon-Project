# RAG Q&A Agent Demo with SpoonOS Integration
# 单文件最小版本，包含向量检索 + LLM 模拟 + SpoonOS Agent

from sentence_transformers import SentenceTransformer
import numpy as np
import spoonos

# ----------------------------
# 向量数据库（最小化实现）
# ----------------------------
class SimpleVectorDB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = []
        self.embeddings = []

    def add_document(self, text):
        self.docs.append(text)
        self.embeddings.append(self.model.encode(text))

    def query(self, query_text, top_k=1):
        q_vec = self.model.encode(query_text)
        scores = [np.dot(q_vec, e)/(np.linalg.norm(q_vec)*np.linalg.norm(e)) for e in self.embeddings]
        idx = np.argmax(scores)
        return self.docs[idx]

# ----------------------------
# 模拟 LLM
# ----------------------------
class SimpleLLM:
    def generate(self, prompt):
        # 模拟生成回答
        return f"【模拟回答】根据文档内容：{prompt}"

# ----------------------------
# RAG Agent
# ----------------------------
class RAGAgent:
    def __init__(self, llm, vector_db: SimpleVectorDB):
        self.llm = llm
        self.vector_db = vector_db

    def answer_question(self, question: str):
        docs = self.vector_db.query(question)
        prompt = f"根据以下内容回答问题：{docs}\n问题：{question}\n回答："
        return self.llm.generate(prompt)

# ----------------------------
# 初始化
# ----------------------------
vector_db = SimpleVectorDB()
# 添加示例文档
vector_db.add_document("AI 最新研究方向包括多模态学习、强化学习优化、生成式模型和可解释 AI。")
vector_db.add_document("RAG 系统结合向量检索和 LLM 输出，提高问答准确性和可解释性。")

llm = SimpleLLM()
rag_agent = RAGAgent(llm, vector_db)

# ----------------------------
# SpoonOS 集成
# ----------------------------
spoonos.api_key = "你的API_KEY"

# 创建 Agent
my_agent = spoonos.Agent.create(
    name="RAG_QnA_Agent",
    type="generative",
    model="llm-model-name"
)

# 绑定任务
def handle_question(task):
    question = task.get("input")
    answer = rag_agent.answer_question(question)
    return {"output": answer}

my_agent.on_task(handle_question)

# ----------------------------
# CLI 测试
# ----------------------------
if __name__ == "__main__":
    print("=== RAG Q&A Agent Demo ===")
    while True:
        question = input("请输入问题: ")
        answer = rag_agent.answer_question(question)
        print(answer)
