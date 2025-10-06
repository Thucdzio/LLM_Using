

---

## Free stack (no OpenAI cost)

**Embeddings:** HuggingFace `BAAI/bge-m3` (local)  
**LLM (optional):**
- **Ollama** (local): `ollama pull llama3:8b-instruct`
- **Groq** (OpenAI-compatible): set `LLM_BACKEND=groq` and `GROQ_API_KEY`

### Install
```bash
pip install -r requirements.txt
```

### Build index (free)
```bash
python src/build_index_free.py --input data/raw --persist vectordb
```

### Classify (optional)
```bash
python src/classify_free.py --query_file data/raw/t.json --backend ollama
```
