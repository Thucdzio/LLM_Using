import argparse
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config_free import VECTORDB_DIR

from config_free import LLM_BACKEND, GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL

load_dotenv()

def get_llm(backend: str):
    if backend == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        return ChatOpenAI(
            model=GROQ_MODEL,
            openai_api_key=api_key,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0
        )
    elif backend == "ollama":
        return ChatOpenAI(
            model=OLLAMA_MODEL,
            openai_api_base="http://localhost:11434/v1",
            openai_api_key="ollama",  # fake key cho ollama
            temperature=0
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

def main(args):
    # load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # load index (FAISS)
    db = Chroma(
    persist_directory=VECTORDB_DIR,
    embedding_function=embeddings
)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(args.query)

    # lấy context
    context = "\n\n".join([d.page_content for d in docs])

    # build prompt
    prompt = f"""Bạn là một trợ lý AI. 
    Trả lời câu hỏi dựa trên ngữ cảnh dưới đây. Nếu không tìm thấy thông tin, hãy nói "Không rõ".

    Ngữ cảnh:
    {context}

    Câu hỏi:
    {args.query}
    """

    # gọi LLM
    llm = get_llm(args.backend or LLM_BACKEND)
    resp = llm.invoke(prompt)
    print("\n=== Kết quả ===\n")
    print(resp.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Câu hỏi muốn tìm kiếm")
    parser.add_argument("--backend", type=str, default=LLM_BACKEND, help="groq hoặc ollama")
    args = parser.parse_args()
    main(args)
