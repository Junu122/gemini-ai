# import fitz  # PyMuPDF
# from sentence_transformers import SentenceTransformer
# import chromadb
# import requests
# import os
# import tempfile
# from dotenv import load_dotenv
# from urllib.parse import urlparse

# load_dotenv()

# # Configuration
# PDF_SOURCE = "client_data.pdf"  # Local path or URL
# CHROMA_DB_PATH = ".chromadb"
# COLLECTION_NAME = "client_knowledge"
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# CHUNK_SIZE, CHUNK_OVERLAP = 500, 100
# GEMINI_MODEL_NAME = "gemini-1.5-pro-latest"

# def get_pdf_content(pdf_source):
#     """Process PDF from URL or local path and extract text"""
#     # Check if URL or local file
#     is_url = bool(urlparse(pdf_source).scheme)
    
#     if is_url:
#         try:
#             print(f"📥 Downloading PDF from {pdf_source}...")
#             response = requests.get(pdf_source, stream=True, timeout=30)
#             response.raise_for_status()
            
#             # Save to temp file
#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
#             temp_path = temp_file.name
#             with open(temp_path, 'wb') as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     if chunk: f.write(chunk)
#             pdf_path = temp_path
#         except Exception as e:
#             print(f"❌ Download error: {str(e)}")
#             return None
#     else:
#         if not os.path.exists(pdf_source):
#             print(f"❌ File not found: {pdf_source}")
#             return None
#         pdf_path = pdf_source
    
#     # Extract text
#     try:
#         print(f"📄 Extracting text from PDF...")
#         doc = fitz.open(pdf_path)
#         text = "".join(page.get_text() for page in doc)
#         doc.close()
        
#         # Cleanup temp file if downloaded
#         if is_url:
#             try: os.unlink(pdf_path)
#             except: pass
            
#         return text
#     except Exception as e:
#         print(f"❌ Text extraction error: {str(e)}")
#         return None

# def process_document(text):
#     """Process document text into embedding database"""
#     # Split text into chunks
#     chunks = [text[i:i + CHUNK_SIZE] 
#               for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)]
#     print(f"   Split into {len(chunks)} chunks")
    
#     # Initialize embedding model
#     print("🧠 Loading embedding model...")
#     model = SentenceTransformer(EMBEDDING_MODEL)
    
#     # Generate embeddings and store in ChromaDB
#     print("💾 Creating vector store...")
#     client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
#     collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
#     # Generate embeddings
#     embeddings = model.encode(chunks, show_progress_bar=True)
#     ids = [f"doc_{i}" for i in range(len(chunks))]
    
#     # Store in database
#     collection.upsert(ids=ids, documents=chunks, embeddings=embeddings.tolist())
#     print(f"✅ Vector store updated with {len(chunks)} chunks")
    
#     return model, collection

# def ask_question(question, embedding_model, collection, api_key):
#     """Answer a question using the document and Gemini"""
#     # Find relevant chunks
#     print("🔍 Finding relevant information...")
#     query_embedding = embedding_model.encode([question])[0].tolist()
#     results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
#     # Check if results found
#     if not results or not results.get("documents") or not results["documents"][0]:
#         return "Could not find relevant information in the document."
    
#     # Combine context chunks
#     context = "\n---\n".join(results["documents"][0])
    
#     # Query Gemini API
#     print("🤖 Generating answer...")
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}"
    
#     prompt = f"""
#     You are answering questions based on specific document content. Answer ONLY using the provided context.
    
#     Context:
#     {context}
    
#     Question:
#     {question}
    
#     Instructions:
#     1. Answer based ONLY on information in the context above.
#     2. If the information is not in the context, respond: "I don't have information about that in the provided documents."
#     3. Be thorough and accurate - include all relevant details from the context.
#     """
    
#     payload = {
#         "contents": [{"parts": [{"text": prompt}]}],
#         "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024}
#     }
    
#     try:
#         response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
#         response.raise_for_status()
#         data = response.json()
        
#         # Extract answer from response
#         if "candidates" in data and data["candidates"]:
#             return data["candidates"][0]["content"]["parts"][0]["text"]
#         else:
#             return "Failed to generate an answer."
#     except Exception as e:
#         return f"Error: {str(e)}"

# def main():
#     # Get API key
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         print("❌ Missing GEMINI_API_KEY in environment variables or .env file")
#         return
    
#     # Get PDF source
#     pdf_source = input("📄 Enter PDF path or URL (press Enter for default): ").strip() or PDF_SOURCE
    
#     # Process PDF
#     text = get_pdf_content(pdf_source)
#     if not text or not text.strip():
#         print("❌ No text extracted from PDF")
#         return
    
#     # Process document
#     embedding_model, collection = process_document(text)
#     print("✅ System ready for questions\n")
    
#     # Q&A loop
#     while True:
#         question = input("❓ Ask a question about the document (or type 'exit'): ").strip()
#         if question.lower() in ['exit', 'quit']:
#             print("👋 Exiting program.")
#             break
#         if not question:
#             continue
            
#         # Answer question
#         answer = ask_question(question, embedding_model, collection, api_key)
#         print(f"\n💡 Answer:\n{answer}\n")

# if __name__ == "__main__":
#     main()