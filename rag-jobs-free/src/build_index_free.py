import os, json, argparse, glob
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from serialize import serialize_record
from config_free import CHUNK_SIZE, CHUNK_OVERLAP, VECTORDB_DIR, HF_EMBEDDING_MODEL

def load_jobs(path):
    paths = []
    if os.path.isdir(path):
        paths = glob.glob(os.path.join(path, "*.json"))
    else:
        paths = [path]

    jobs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "jobs" in data:
                jobs.extend(data["jobs"])
            elif isinstance(data, list):
                jobs.extend(data)
    return jobs

def main(args):
    load_dotenv()
    jobs = load_jobs(args.input)
    print(f"Loaded {len(jobs)} job records")

    docs = []
    for j in jobs:
        text, meta = serialize_record(j)
        docs.append(Document(page_content=text, metadata=meta))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size or CHUNK_SIZE,
        chunk_overlap=args.chunk_overlap or CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=args.persist or VECTORDB_DIR
    )
    vectordb.add_documents(chunks)
    vectordb.persist()
    print(f"Indexed {len(chunks)} chunks into {args.collection} @ {args.persist or VECTORDB_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--persist", default=None)
    parser.add_argument("--collection", default="jobs_vi")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--chunk_overlap", type=int, default=None)
    args = parser.parse_args()
    main(args)
