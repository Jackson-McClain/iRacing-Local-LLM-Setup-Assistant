print("DEBUG: Script started. Loading imports...")
import pandas as pd
import os
from operator import itemgetter
print("DEBUG: Core libraries imported.")

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- UPDATED IMPORTS ---
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
# -------------------------
print("DEBUG: LangChain imports loaded.")

# --- CONFIGURATION ---
OLLAMA_MODEL = "deepseek-r1"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"
DB_PATH = "./chromadb"
# ----------------------

# Initialize Ollama LLM and embeddings
print("DEBUG: Initializing OllamaLLM...")
ollama_model = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
print("DEBUG: OllamaLLM initialized.")

print("DEBUG: Initializing Embeddings...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
print("DEBUG: Embeddings initialized.")

# Load or create Chroma vector store
print("DEBUG: Loading Chroma database...")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
print("DEBUG: Chroma database loaded.")

retriever = db.as_retriever(search_kwargs={"k": 3})
print("DEBUG: Retriever created.")


def analyze_telemetry(file_path: str) -> str:
    """Summarize key telemetry metrics from CSV."""
    if not file_path:
        return "No telemetry data provided."
    try:
        df = pd.read_csv(file_path)
        summary = f"Telemetry Summary for {os.path.basename(file_path)}:\n"
        for col in df.columns:
            if any(keyword in col for keyword in ["Temp", "Shock", "Travel"]):
                summary += f"{col}: avg={df[col].mean():.2f}, max={df[col].max():.2f}\n"
        return summary
    except Exception as e:
        return f"Error reading telemetry: {e}"

def format_docs(docs):
    """Helper function to format retrieved documents into a string."""
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# --- Define the Modern RAG Chain ---
print("DEBUG: Defining RAG chain...")
template = """
You are an iRacing Dirt Midget setup engineer with professional-level expertise in tuning and driving these cars.

Your task is to give *only short, numerical, and specific setup adjustments* that match the driver’s question. 
You may only use parameters explicitly listed in the setup documentation below — **no exceptions**.
If a parameter or concept (e.g. gas pressure, CG height, right-side offset) is not listed, you must not reference it.

Follow these strict output rules:
- Exactly 1 to 3 bullet points maximum
- Each bullet must include: the corner (e.g. RF, LR), the part (e.g. spring, shock, torsion bar), and the precise numerical adjustment amount (e.g. “Add 10 lbs to RR spring”)
- Every change must be quantitative (include a number and unit or clicks)
- No phrases like “increase,” “reduce,” “consider,” or “potentially”
- No explanations, context, or summaries — just the adjustments
- Do not reference any parameters outside the setup documentation

Setup documentation (allowed parameters only):
---
{context}
---

Telemetry summary:
---
{telemetry}
---

Driver question:
"{question}"

Respond with only 1–3 bullet points formatted exactly like this example:

- Add 0.5 rebound clicks to LR shock  
- Soften RR spring by 10 lbs  
- Increase stagger by 0.25 inches  
"""
prompt = PromptTemplate.from_template(template)


rag_chain = (
    RunnablePassthrough.assign(
        # FIX: Use itemgetter("question") to properly extract the string
        # in an LCEL-compatible way before piping it to the retriever.
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt
    | ollama_model
    | StrOutputParser()
)
print("DEBUG: RAG chain defined.")


def get_setup_advice(question: str, telemetry_file: str = None) -> str:
    """Prepare data and invoke the RAG chain."""
    telemetry_summary = analyze_telemetry(telemetry_file)
    
    input_data = {
        "question": question,
        "telemetry": telemetry_summary
    }
    
    return rag_chain.invoke(input_data)


if __name__ == "__main__":
    print("iRacing Setup Assistant started! (Using Modern LCEL)\n---") # Added a visual separator
    while True:
        # We removed all emojis to prevent crashes
        question = input("\nAsk your setup question (or 'quit'): ")
        if question.lower() in ["quit", "exit"]:
            break
        telemetry_path = input("Optional telemetry CSV path (leave blank if none): ").strip() or None
        
        print("\nThinking...\n")
        answer = get_setup_advice(question, telemetry_path)
        print(f"Setup Advice: {answer}\n")
