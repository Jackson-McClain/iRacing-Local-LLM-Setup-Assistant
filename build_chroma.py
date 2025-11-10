from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import os

# --- Config ---
EMBED_MODEL = "nomic-embed-text"  # Make sure this model is pulled via Ollama
OLLAMA_URL = "http://localhost:11434"
DOCS_PATH = "./setup_docs"
DB_PATH = "./chromadb"
# --------------

def build_chroma():
    print(f"Loading documents from {DOCS_PATH}...")
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    if not docs:
        print(f"No documents found in {DOCS_PATH}. Aborting.")
        return

    print(f"Loaded {len(docs)} documents. Initializing embeddings...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

    print("Building Chroma database (this may take a moment)...")
    # ✅ Fix: Pass the 'embeddings' object here
    db = Chroma.from_documents(
        docs, 
        embeddings,  # This is the crucial fix
        persist_directory=DB_PATH
    )
    
    db.persist()
    print(f"✅ Database built successfully at {DB_PATH} with {len(docs)} documents.")

if __name__ == "__main__":
    if not os.path.exists(DOCS_PATH) or not os.listdir(DOCS_PATH):
        print(f"Error: Document path '{DOCS_PATH}' is empty or does not exist.")
        print("Please create the folder and add your .txt setup guides.")
    else:
        # We must ensure the DB_PATH exists for the persist() call
        os.makedirs(DB_PATH, exist_ok=True)
        build_chroma()