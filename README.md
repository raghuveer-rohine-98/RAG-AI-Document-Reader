# 🤖 RAG AI Document Reader

A simple **Retrieval-Augmented Generation (RAG)** application built using **LangChain**, **Chroma**, and **Gradio**.
This tool lets you **upload documents and chat with them**, powered by **OpenAI** or **local Ollama models** — all running on your own machine.

---

## 🪄 What You Can Do

* 📄 Upload your PDFs, text, or markdown files
* 💬 Ask questions about the uploaded content
* 🧠 Choose between **OpenAI (cloud)** or **Ollama (local)** models
* ⚡ Automatically saves embeddings in **ChromaDB** for fast retrieval
* 🚫 Avoids duplicate file re-indexing automatically
* 🧹 Reset the vector database anytime

---

## ⚙️ Prerequisites

You only need:

* 🐍 **Python 3.10+**
* 🌐 Internet connection (for installing dependencies or using OpenAI)
* 🧩 (Optional) [Ollama](https://ollama.com/) if you want to run models locally

---

## 🚀 Getting Started (For Complete Beginners)

Follow these steps **exactly** — even if you’ve never set up Python before 👇

---

### **1️⃣ Download or Clone the Repository**

If you have Git installed:

```bash
git clone https://github.com/raghuveer-rohine-98/RAG-AI-Document-Reader.git
cd rag-ai-doc-reader
```

Or download ZIP manually from GitHub → extract it → open the folder in terminal.

---

### **2️⃣ Check Python Installation**

Run this:

```bash
python --version
```

If it shows something like `Python 3.10.x` or higher, you’re good.
If not → [Download Python](https://www.python.org/downloads/) and install it.

---

### **3️⃣ Create a Virtual Environment**

```bash
python -m venv venv
```

Activate it:

* **Windows:**

  ```bash
  venv\Scripts\activate
  ```
* **macOS / Linux:**

  ```bash
  source venv/bin/activate
  ```

You’ll know it’s active when your terminal prompt starts with `(venv)`.

---

### **4️⃣ Install All Dependencies**

Run this (you only need internet for this step):

```bash
pip install -r requirements.txt
```

> 💡 If you don’t have a `requirements.txt`, just run this instead:
>
> ```bash
> pip install langchain langchain-openai langchain-chroma langchain-community gradio requests
> ```

---

### **5️⃣ (Optional) Set Up Your LLM Provider**

#### 👉 Option 1: Use **OpenAI API**

You’ll need an OpenAI API key from [platform.openai.com](https://platform.openai.com/account/api-keys).

Then run this:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

(Use `set` instead of `export` on Windows.)

#### 👉 Option 2: Use **Ollama (local models)**

1. Install Ollama: [https://ollama.com/download](https://ollama.com/download)
2. Run it:

   ```bash
   ollama serve
   ```
3. Optionally pull a model:

   ```bash
   ollama pull llama3
   ```

---

### **6️⃣ Run the App**

Once everything’s installed, simply run:

```bash
python rag_app.py
```

After a few seconds, you’ll see:

```
Running on http://127.0.0.1:7860
```

Open that link in your browser — the app UI will appear 🎉

---

### **7️⃣ Using the App**

1. Go to the **Configuration** tab

   * Select “OpenAI” or “Ollama”
   * Enter API key (if OpenAI) or select model (if Ollama)
   * Click **Save Configuration**

2. Go to the **Load Documents** tab

   * Upload `.pdf`, `.txt`, or `.md` files
   * Wait until you see “✅ Loaded ... chunks”
   * You can upload more files later — they’ll be appended automatically

3. Go to the **Chat** tab

   * Ask questions like:

     > “Summarize the document.”
     > “What are the main findings in the PDF?”
     > “Compare the resumes.”

---

### **8️⃣ (Optional) Reset the Vector DB**

If you want to clear all embeddings and start fresh:

* Click the **🗑️ Reset Vector DB** button in the “Load Documents” tab
  or
* Run this in terminal:

  ```bash
  rm -rf temp_vector_db uploaded_files
  ```

---

## 🧠 Tech Stack

* **LangChain** – for RAG pipeline
* **Chroma** – for vector storage
* **Gradio** – for the web UI
* **OpenAI / Ollama** – for LLM responses

---

## 🪄 Example Use Cases

* Summarize research papers
* Extract insights from long reports
* Compare resumes or documents
* Study NCERT or reference notes interactively

---

## 📘 License

This project is open source under the **MIT License**.

---

## ✨ Author

**Raghuveer Rohine**
💡 AI + Spring Boot + LangChain Enthusiast
📫 [GitHub Profile](https://github.com/raghuveer-rohine)
