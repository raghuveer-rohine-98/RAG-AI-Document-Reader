import os
import hashlib
import shutil
from pathlib import Path
from typing import List
import requests
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings


class LocalRAGApp:
    def __init__(self):
        self.vectorstore = None
        self.conversation_chain = None
        self.memory = None
        self.current_config = {}
        self.documents_loaded = False

    # ---------------- Ollama Connection ----------------
    def check_ollama_connection(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return True, [model["name"] for model in models]
            return False, []
        except Exception:
            return False, []

    # ---------------- LLM Setup ----------------
    def setup_llm_and_embeddings(self, provider, api_key=None, model=None, temperature=0.7):
        try:
            if provider == "OpenAI":
                if not api_key:
                    return False, "OpenAI API key is required"
                os.environ["OPENAI_API_KEY"] = api_key
                self.current_config = {
                    "provider": provider,
                    "model": model,
                    "temperature": temperature,
                    "api_key": api_key,
                }
                return True, f"OpenAI configured with model: {model}"

            elif provider == "Ollama":
                is_running, available_models = self.check_ollama_connection()
                if not is_running:
                    return False, "Ollama is not running. Please start Ollama first."
                if model not in available_models:
                    return False, f"Model {model} not available in Ollama"

                self.current_config = {
                    "provider": provider,
                    "model": model,
                    "temperature": temperature,
                }
                return True, f"Ollama configured with model: {model}"

        except Exception as e:
            return False, f"Error configuring LLM: {str(e)}"

    # ---------------- Document Loading ----------------
    def load_documents(self, files: List, file_types: List[str]) -> tuple:
        """Load and append new (non-duplicate) documents into Chroma vectorstore"""
        try:
            if not files:
                return False, "No files selected", 0

            # Temporary upload directory
            upload_dir = "uploaded_files"
            os.makedirs(upload_dir, exist_ok=True)

            # Copy uploaded files to a local directory
            file_paths = []
            for f in files:
                target_path = os.path.join(upload_dir, os.path.basename(f.name))
                shutil.copy(f.name, target_path)
                file_paths.append(target_path)

            # Loader map
            loaders_map = {
                ".pdf": PyPDFLoader,
                ".txt": TextLoader,
                ".md": UnstructuredMarkdownLoader,
            }

            documents = []
            total_files = 0
            skipped_files = 0
            db_path = "temp_vector_db"

            # Choose embeddings
            if self.current_config["provider"] == "OpenAI":
                embeddings = OpenAIEmbeddings()
            else:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Load existing DB if available
            if os.path.exists(db_path) and os.listdir(db_path):
                self.vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
                try:
                    existing_metadatas = self.vectorstore.get(include=["metadatas"])["metadatas"]
                    existing_hashes = {m.get("hash") for m in existing_metadatas if "hash" in m}
                except Exception:
                    existing_hashes = set()
            else:
                self.vectorstore = None
                existing_hashes = set()

            # Process files
            for file_path in file_paths:
                ext = os.path.splitext(file_path)[1].lower()
                if ext not in loaders_map or ext not in file_types:
                    continue

                total_files += 1

                # Compute file hash
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in existing_hashes:
                    skipped_files += 1
                    print(f"Skipping duplicate: {file_path}")
                    continue

                loader = loaders_map[ext](file_path)
                file_docs = loader.load()

                for doc in file_docs:
                    doc.metadata.update(
                        {
                            "source": str(file_path),
                            "file_type": ext,
                            "filename": os.path.basename(file_path),
                            "hash": file_hash,
                        }
                    )

                documents.extend(file_docs)

            if not documents:
                return False, f"No new documents to add (skipped {skipped_files} duplicates)", 0

            # Split into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            # Append or create new
            if self.vectorstore is None:
                print("Creating new Chroma DB...")
                self.vectorstore = Chroma.from_documents(
                    documents=chunks, embedding=embeddings, persist_directory=db_path
                )
            else:
                print("Appending to existing Chroma DB...")
                self.vectorstore.add_documents(chunks)

            # Safe persist (only if available)
            if hasattr(self.vectorstore, "persist"):
                self.vectorstore.persist()

            # Setup LLM if needed
            if not self.conversation_chain:
                if self.current_config["provider"] == "OpenAI":
                    llm = ChatOpenAI(
                        temperature=self.current_config["temperature"],
                        model_name=self.current_config["model"],
                        api_key=self.current_config["api_key"],
                    )
                else:
                    llm = ChatOpenAI(
                        temperature=self.current_config["temperature"],
                        model_name=self.current_config["model"],
                        base_url="http://localhost:11434/v1",
                        api_key="ollama",
                    )

                self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                retriever = self.vectorstore.as_retriever()
                self.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm, retriever=retriever, memory=self.memory
                )

            self.documents_loaded = True
            added_files = total_files - skipped_files
            return True, f"✅ Added {added_files} new files, skipped {skipped_files} duplicates ({len(chunks)} chunks)", len(chunks)

        except Exception as e:
            return False, f"Error loading documents: {str(e)}", 0

    # ---------------- Chat ----------------
    def chat(self, message: str, history: List) -> tuple:
        try:
            if not self.conversation_chain:
                return history, "Please load documents first"
            result = self.conversation_chain.invoke({"question": message})
            answer = result["answer"]
            history.append([message, answer])
            return history, ""
        except Exception as e:
            history.append([message, f"Error processing question: {str(e)}"])
            return history, ""


# ---------------- Gradio Interface ----------------
app = LocalRAGApp()


def create_interface():
    with gr.Blocks(title="Local RAG Chat Application") as demo:
        gr.Markdown("# 🤖 Local RAG Chat Application")
        gr.Markdown("Chat with your uploaded documents using OpenAI or local Ollama")

        with gr.Tabs():
            # --- Configuration ---
            with gr.TabItem("⚙️ Configuration"):
                provider_radio = gr.Radio(choices=["OpenAI", "Ollama"], label="LLM Provider", value="OpenAI")

                with gr.Group(visible=True) as openai_config:
                    api_key_input = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")
                    openai_model = gr.Dropdown(
                        choices=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini"],
                        label="Model",
                        value="gpt-4o-mini",
                    )
                    openai_temp = gr.Slider(0.0, 2.0, 0.7, 0.1, label="Temperature")

                with gr.Group(visible=False) as ollama_config:
                    ollama_status = gr.Textbox(label="Ollama Status", interactive=False)
                    refresh_models_btn = gr.Button("🔄 Refresh Models")
                    ollama_model = gr.Dropdown(choices=[], label="Available Models", interactive=False)
                    ollama_temp = gr.Slider(0.0, 2.0, 0.7, 0.1, label="Temperature")

                config_btn = gr.Button("💾 Save Configuration", variant="primary")
                config_status = gr.Textbox(label="Configuration Status", interactive=False)

                def toggle_provider(provider):
                    return gr.update(visible=provider == "OpenAI"), gr.update(visible=provider == "Ollama")

                def check_ollama_status():
                    is_running, models = app.check_ollama_connection()
                    if is_running:
                        return "✅ Ollama is running", gr.update(choices=models, interactive=True)
                    else:
                        return "❌ Ollama is not running", gr.update(choices=[], interactive=False)

                def save_configuration(provider, api_key, openai_model_val, openai_temp_val, ollama_model_val, ollama_temp_val):
                    if provider == "OpenAI":
                        success, message = app.setup_llm_and_embeddings(provider, api_key, openai_model_val, openai_temp_val)
                    else:
                        success, message = app.setup_llm_and_embeddings(provider, None, ollama_model_val, ollama_temp_val)
                    return "✅ " + message if success else "❌ " + message

                provider_radio.change(toggle_provider, inputs=[provider_radio], outputs=[openai_config, ollama_config])
                refresh_models_btn.click(check_ollama_status, outputs=[ollama_status, ollama_model])
                config_btn.click(save_configuration, inputs=[provider_radio, api_key_input, openai_model, openai_temp,
                                                             ollama_model, ollama_temp], outputs=[config_status])

                demo.load(check_ollama_status, outputs=[ollama_status, ollama_model])

            # --- Load Documents ---
            with gr.TabItem("📁 Load Documents"):
                gr.Markdown("## Step 2: Upload your documents")

                file_upload = gr.File(label="Upload Files", file_count="multiple", file_types=[".pdf", ".txt", ".md"])
                file_types = gr.CheckboxGroup(choices=[".pdf", ".txt", ".md"], label="File Types", value=[".pdf", ".txt", ".md"])
                load_btn = gr.Button("📚 Load Documents", variant="primary")
                reset_btn = gr.Button("🗑️ Reset Vector DB")
                load_status = gr.Textbox(label="Loading Status", interactive=False)

                def load_documents_from_upload(files, selected_types):
                    success, message, _ = app.load_documents(files, selected_types)
                    return message

                def reset_vector_db():
                    db_path = "temp_vector_db"
                    if os.path.exists(db_path):
                        shutil.rmtree(db_path)
                    return "✅ Vector DB reset successfully."

                load_btn.click(load_documents_from_upload, inputs=[file_upload, file_types], outputs=[load_status])
                reset_btn.click(reset_vector_db, outputs=[load_status])

            # --- Chat ---
            with gr.TabItem("💬 Chat"):
                chatbot = gr.Chatbot(height=500, label="Chat History")
                msg = gr.Textbox(label="Your Question", placeholder="Ask about your documents...")
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("🧹 Clear Chat")

                def respond(message, history):
                    if not app.documents_loaded:
                        return history + [[message, "❌ Please load documents first."]], ""
                    return app.chat(message, history)

                def clear_history():
                    if app.memory:
                        app.memory.clear()
                    return []

                msg.submit(respond, [msg, chatbot], [chatbot, msg])
                send_btn.click(respond, [msg, chatbot], [chatbot, msg])
                clear_btn.click(clear_history, outputs=[chatbot])

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)


