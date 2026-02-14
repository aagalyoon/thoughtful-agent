import os
import json
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

KNOWLEDGE_BASE = [
    {
        "question": "What does the eligibility verification agent (EVA) do?",
        "answer": "EVA automates the process of verifying a patient's eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections.",
    },
    {
        "question": "What does the claims processing agent (CAM) do?",
        "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements.",
    },
    {
        "question": "How does the payment posting agent (PHIL) work?",
        "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden.",
    },
    {
        "question": "Tell me about Thoughtful AI's Agents.",
        "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others.",
    },
    {
        "question": "What are the benefits of using Thoughtful AI's agents?",
        "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting.",
    },
]

SIMILARITY_THRESHOLD = 0.35


class SupportAgent:
    def __init__(self):
        self.questions = [item["question"] for item in KNOWLEDGE_BASE]
        self.answers = [item["answer"] for item in KNOWLEDGE_BASE]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            try:
                from anthropic import Anthropic

                self._llm_client = Anthropic()
            except Exception:
                self._llm_client = False
        return self._llm_client

    def find_best_match(self, user_input: str) -> tuple[str | None, float]:
        input_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(input_vector, self.question_vectors)[0]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= SIMILARITY_THRESHOLD:
            return self.answers[best_idx], best_score
        return None, best_score

    def llm_fallback(self, user_input: str, history: list) -> str:
        if not self.llm_client:
            return (
                "I don't have specific information about that in my knowledge base. "
                "I can help with questions about Thoughtful AI's agents like EVA "
                "(Eligibility Verification), CAM (Claims Processing), and PHIL "
                "(Payment Posting). What would you like to know?"
            )

        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        response = self.llm_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=512,
            system=(
                "You are a helpful customer support agent for Thoughtful AI, a company "
                "that builds AI-powered automation agents for healthcare revenue cycle "
                "management. Their main products are EVA (Eligibility Verification), "
                "CAM (Claims Processing), and PHIL (Payment Posting). Be concise and helpful."
            ),
            messages=messages,
        )
        return response.content[0].text

    def respond(self, user_input: str, history: list) -> str:
        if not user_input.strip():
            return "Please enter a question and I'll do my best to help."

        answer, score = self.find_best_match(user_input)

        if answer:
            return answer

        return self.llm_fallback(user_input, history)


agent = SupportAgent()


def chat(message: str, history: list[dict]) -> str:
    return agent.respond(message, history)


demo = gr.ChatInterface(
    fn=chat,
    title="Thoughtful AI Support",
    description="Ask me about Thoughtful AI's automation agents for healthcare.",
    examples=[
        "What does EVA do?",
        "Tell me about Thoughtful AI's agents",
        "How does payment posting work?",
        "What are the benefits of using your agents?",
    ],
)

if __name__ == "__main__":
    demo.launch()
