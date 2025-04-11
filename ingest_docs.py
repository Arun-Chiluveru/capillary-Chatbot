import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_markdown(docs_path="docs", vectorstore_dir="vectorstore"):
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Prepare text chunks list
    all_chunks = []

    # Ensure docs folder exists
    if not os.path.exists(docs_path):
        print(f"❌ Directory '{docs_path}' does not exist.")
        return

    # Iterate over .md files
    for filename in os.listdir(docs_path):
        if filename.endswith(".md"):
            filepath = os.path.join(docs_path, filename)
            try:
                print(f"📄 Processing: {filename}")
                loader = TextLoader(filepath, encoding="utf-8")
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(documents)
                all_chunks.extend(chunks)

                print(f"✅ Added {len(chunks)} chunks from {filename}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    if not all_chunks:
        print("⚠️ No text chunks were generated. Check your markdown files.")
        return

    # Create FAISS vector store
    print(f"\n🔧 Creating FAISS vectorstore with {len(all_chunks)} chunks...")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    # Save vector store
    vectorstore.save_local(vectorstore_dir)
    print(f"✅ Vectorstore saved to: {vectorstore_dir}")

if __name__ == "__main__":
    ingest_markdown()
