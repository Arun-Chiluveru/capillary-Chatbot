import os
from typing import Optional, List, Any

os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
st.set_page_config(page_title="Capillarytech Chatbot", layout="wide")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms.base import LLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Constants ---
MODEL_NAME = "google/flan-t5-base"
VECTORSTORE_PATH = "vectorstore_docs"

# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# --- Custom Transformers Wrapper ---
class TransformersLLM(LLM):
    model: Any
    tokenizer: Any

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "transformers"

llm = TransformersLLM(model=model, tokenizer=tokenizer)

# --- UI ---
st.title("🤖 Capillarytech Chatbot")
st.caption("Ask a question based on your Capillary documents")

# --- Load FAISS Vectorstore ---
if not os.path.exists(f"{VECTORSTORE_PATH}/index.faiss") or not os.path.exists(f"{VECTORSTORE_PATH}/index.pkl"):
    st.error("❌ Vectorstore not found. Please run `ingest_docs.py` first.")
    st.stop()

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local(VECTORSTORE_PATH, embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# --- Input ---
query = st.text_input("💬 Ask a question:")

if query:
    with st.spinner("Fetching..."):
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content[:500] for doc in docs[:3])

        with st.expander("📄 Retrieved Context (debug only)"):
            st.write(context)

        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}
Answer:"""

        answer = llm(prompt)
        st.markdown("### 🤖 Answer")
        st.write(answer.strip())
