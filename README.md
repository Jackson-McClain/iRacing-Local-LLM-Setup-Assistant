# 🏎️ iRacing Setup Assistant

An intelligent dirt midget setup assistant powered by **LangChain**, **Ollama**, and **Chroma DB**.  
This tool analyzes driver questions and optional telemetry data to recommend **specific, numerical iRacing setup adjustments**.

---

## 🚀 Features
- Accepts natural language driver feedback (e.g. “Car is loose on entry”)
- Provides **numerical setup tweaks** like:
  - “Add 0.5 rebound to RR shock”
  - “Reduce stagger by 0.25 inches”
- Uses **retrieval-augmented generation (RAG)** with setup documentation
- Optionally analyzes telemetry CSV files for average/max shock or tire values
- Fully local (no API keys or internet required)

---

## 🧠 How It Works
1. Setup documents and telemetry data are embedded using `nomic-embed-text`
2. A retriever (Chroma) finds relevant setup context for your question
3. The LLM (DeepSeek-R1 via Ollama) generates short, precise setup advice

---

## 🧩 Requirements
- Python 3.10+
- Ollama (running locally with model `deepseek-r1`)
- Chroma and LangChain libraries
