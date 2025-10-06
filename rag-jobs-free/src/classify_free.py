import os, argparse, json
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
from config_free import LLM_BACKEND, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL
print("DEBUG: GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))

def get_llm(backend: str):
    backend = backend.lower()
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)
    elif backend == "groq":
        from langchain_openai import ChatOpenAI
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        return ChatOpenAI(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    raise RuntimeError("Unsupported backend (use ollama|groq)")

TAXONOMY = {
    "industry": ["Manufacturing","Technology","Finance","Healthcare","Retail","Logistics","Education","Media","Beauty & Personal Care"],
    "role_family": ["Sales","Marketing","Operations","Engineering","Product","Finance","HR","Customer Success","Supply Chain","eCommerce"]
}

PROMPT_TMPL = ChatPromptTemplate.from_template(
    """Bạn là hệ thống phân loại tin tuyển dụng Việt Nam.
Nhiệm vụ: dự đoán industry, role_family, canonical_title theo TAXONOMY.
Chỉ sử dụng nội dung tin và kiến thức nghề nghiệp phổ biến.
Nếu không đủ dữ liệu, để "unknown". Trả về JSON đúng schema.

### TAXONOMY
{taxonomy}

### TIN
{job_text}

### YÊU CẦU JSON
{{
  "industry": "...",
  "role_family": "...",
  "canonical_title": "...",
  "confidence_reason": "1-2 câu"
}}
"""
)

def load_job_text(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    j = data["jobs"][0]
    parts = [
        f"[TITLE] {j.get('name','')}",
        f"[COMPANY] {j.get('company','')} | [LOC] {', '.join(j.get('locations', [])[:1])}",
        "[DESCRIPTION]",
        j.get("description",""),
        "[REQUIREMENTS]",
        j.get("requirements",""),
        f"[SKILLS] {j.get('skill','')}"
    ]
    return "\n".join(parts)

def main(args):
    load_dotenv()
    llm = get_llm(args.backend or LLM_BACKEND)
    job_text = load_job_text(args.query_file)
    prompt = PROMPT_TMPL.format_messages(taxonomy=json.dumps(TAXONOMY, ensure_ascii=False), job_text=job_text)
    resp = llm.invoke(prompt)
    print(resp.content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--backend", default=None)
    args = parser.parse_args()
    main(args)
# python src/classify_free.py --query_file data/raw/t.json --backend groq