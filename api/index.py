from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import os
import tempfile
import random
import threading
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from urllib.parse import urlparse
from werkzeug.middleware.proxy_fix import ProxyFix

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Apply ProxyFix to handle reverse proxy headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configure CORS properly for production
# Change this to your actual allowed origins in production
allowed_origins = os.getenv('ALLOWED_ORIGINS', '*')
if allowed_origins != '*':
    allowed_origins = allowed_origins.split(',')
CORS(app, resources={r"/*": {"origins": allowed_origins}})

# Configuration
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', '.chromadb')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-pro-latest')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'app.log')

# Set up logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
handler = RotatingFileHandler(LOG_FILE, maxBytes=10000000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(log_level)

# Global variables
embedding_model = None
client_data = {}
model_lock = threading.Lock()

class ClientData:
    def __init__(self, client_id):
        self.client_id = client_id
        self.collection = None
        self.loaded_documents = []
        self.last_used_time = time.time()
        self.processing = False
        self.document_urls = set()

def initialize_existing_clients():
    """Load all existing client collections at startup with proper URL tracking"""
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collections = chroma_client.list_collections()
        
        for collection in collections:
            if collection.name.startswith("client_"):
                client_id = collection.name.replace("client_", "").replace("_knowledge", "")
                
                # Initialize client data structure
                if client_id not in client_data:
                    client_data[client_id] = ClientData(client_id)
                
                client_info = client_data[client_id]
                client_info.collection = collection
                
                # Get all documents with metadata
                results = collection.get(include=["metadatas"])
                
                # Rebuild document tracking
                doc_map = {}
                if results and results.get("metadatas"):
                    for meta in results["metadatas"]:
                        if meta and meta.get("document_id"):
                            doc_id = meta["document_id"]
                            if doc_id not in doc_map:
                                doc_map[doc_id] = {
                                    "name": meta.get("document_name", "Unknown"),
                                    "url": meta.get("document_url", ""),
                                    "chunk_count": 0,
                                    "upload_time": meta.get("upload_time", time.time())
                                }
                            doc_map[doc_id]["chunk_count"] += 1
                
                # Rebuild client_info structures
                for doc_id, doc_info in doc_map.items():
                    client_info.loaded_documents.append({
                        "id": doc_id,
                        "name": doc_info["name"],
                        "url": doc_info["url"],
                        "source_type": "url" if doc_info["url"] else "upload",
                        "chunks": doc_info["chunk_count"],
                        "upload_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(doc_info["upload_time"]))
                    })
                    if doc_info["url"]:
                        client_info.document_urls.add(doc_info["url"])
                
        app.logger.info(f"Initialized {len(collections)} client collections from persistent storage")
    except Exception as e:
        app.logger.error(f"Error initializing existing clients: {str(e)}")

def get_pdf_content(pdf_source):
    """Process PDF from URL or local path and extract text"""
    is_url = bool(urlparse(pdf_source).scheme)
    
    if is_url:
        try:
            app.logger.info(f"Downloading PDF from {pdf_source}...")
            response = requests.get(pdf_source, stream=True, timeout=30)
            response.raise_for_status()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_path = temp_file.name
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
            pdf_path = temp_path
        except Exception as e:
            app.logger.error(f"Download error: {str(e)}")
            return None
    else:
        if not os.path.exists(pdf_source):
            app.logger.error(f"File not found: {pdf_source}")
            return None
        pdf_path = pdf_source
    
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if is_url:
            try: os.unlink(pdf_path)
            except: pass
        return text
    except Exception as e:
        app.logger.error(f"Text extraction error: {str(e)}")
        return None

def init_embedding_model():
    global embedding_model
    with model_lock:  # Thread safety for model initialization
        if embedding_model is None:
            app.logger.info("Loading embedding model...")
            try:
                embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                app.logger.info("Embedding model loaded successfully")
            except Exception as e:
                app.logger.error(f"Failed to load embedding model: {str(e)}")
                raise

def get_client_collection(client_id):
    """Get or create a ChromaDB collection for a specific client"""
    if client_id not in client_data:
        client_data[client_id] = ClientData(client_id)
    
    client_info = client_data[client_id]
    client_info.last_used_time = time.time()
    
    if client_info.collection is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client_info.collection = chroma_client.get_or_create_collection(name=f"client_{client_id}_knowledge")
        
        # Load existing documents from ChromaDB
        existing_docs = client_info.collection.get(include=["metadatas"])
        if existing_docs and existing_docs.get("metadatas"):
            unique_docs = {}
            for meta in existing_docs["metadatas"]:
                if meta and meta.get("document_id"):
                    doc_id = meta["document_id"]
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = {
                            "name": meta.get("document_name", "Unknown"),
                            "url": meta.get("document_url", ""),
                            "chunk_count": 0
                        }
                    unique_docs[doc_id]["chunk_count"] += 1
            
            for doc_id, doc_info in unique_docs.items():
                client_info.loaded_documents.append({
                    "id": doc_id,
                    "name": doc_info["name"],
                    "url": doc_info["url"],
                    "source_type": "url" if doc_info["url"] else "upload",
                    "chunks": doc_info["chunk_count"],
                    "upload_date": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                if doc_info["url"]:
                    client_info.document_urls.add(doc_info["url"])
    
    return client_info.collection

def process_document(text, client_id, document_name, document_url=None):
    init_embedding_model()
    collection = get_client_collection(client_id)
    client_info = client_data[client_id]
    client_info.processing = True
    
    try:
        document_id = str(uuid.uuid4())
        chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)]
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)  # Progress bar not needed in production
        ids = [f"doc_{document_id}_{i}" for i in range(len(chunks))]
        
        metadatas = [{
            "document_id": document_id,
            "document_name": document_name,
            "document_url": document_url or "",
            "chunk_index": i,
            "upload_time": time.time()
        } for i in range(len(chunks))]
        
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        doc_info = {
            "id": document_id,
            "name": document_name,
            "url": document_url or "",
            "source_type": "url" if document_url else "upload",
            "chunks": len(chunks),
            "upload_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        client_info.loaded_documents.append(doc_info)
        if document_url:
            client_info.document_urls.add(document_url)
        app.logger.info(f"Document {document_id} processed successfully for client {client_id}")
        return document_id
    except Exception as e:
        app.logger.error(f"Processing error for client {client_id}: {str(e)}")
        return None
    finally:
        client_info.processing = False

def ask_question(question, client_id, api_key, document_id=None):
    if client_id not in client_data:
        return {"error": f"Client {client_id} not found. Please upload a document first."}
    
    client_info = client_data[client_id]
    collection = client_info.collection
    client_info.last_used_time = time.time()
    init_embedding_model()
    
    query_embedding = embedding_model.encode([question])[0].tolist()
    query_filter = {"document_id": document_id} if document_id else None
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=5,
            where=query_filter
        )
    except Exception as e:
        app.logger.error(f"Query error for client {client_id}: {str(e)}")
        return {"error": f"Error querying your documents: {str(e)}"}
    
    if not results or not results.get("documents") or not results["documents"][0]:
        return {"error": "Could not find relevant information in your documents."}
    
    doc_names = []
    doc_urls = []
    if "metadatas" in results and results["metadatas"][0]:
        doc_names = [meta.get("document_name", "Unknown") for meta in results["metadatas"][0]]
        doc_urls = [meta.get("document_url", "") for meta in results["metadatas"][0]]
    
    context_chunks = results["documents"][0]
    context_with_sources = []
    
    for i, chunk in enumerate(context_chunks):
        source = doc_names[i] if i < len(doc_names) else "Unknown"
        source_url = doc_urls[i] if i < len(doc_urls) else ""
        source_info = f"Source: {source}"
        if source_url:
            source_info += f" (URL: {source_url})"
        context_with_sources.append(f"{source_info}\nContent: {chunk}")
    
    context = "\n---\n".join(context_with_sources)
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}"
    
    prompt = f"""
    You are answering questions based on specific document content. Answer ONLY using the provided context.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Instructions:
    1. Answer based ONLY on information in the context above.
    2. If the information is not in the context, respond: "I don't have information about that in the provided documents."
    3. Be thorough and accurate - include all relevant details from the context.
    4. When appropriate, mention which document(s) the information comes from.
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024}
    }
    
    max_retries = 5
    response = None
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            if response and response.status_code == 429 and attempt < max_retries - 1:
                backoff_time = (2 ** attempt) + (random.random() * 0.5)
                time.sleep(backoff_time)
            else:
                if response:
                    app.logger.error(f"Gemini API error: {response.status_code} for client {client_id}")
                    return {"error": f"Gemini API error: {response.status_code}"}
                else:
                    app.logger.error(f"Request failed for client {client_id}: {str(e)}")
                    return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            app.logger.error(f"Error with Gemini API for client {client_id}: {str(e)}")
            return {"error": f"Error with Gemini API: {str(e)}"}
    
    try:
        data = response.json()
        if "candidates" in data and data["candidates"]:
            return {"answer": data["candidates"][0]["content"]["parts"][0]["text"]}
        else:
            app.logger.error(f"Failed to generate answer from Gemini for client {client_id}")
            return {"error": "Failed to generate an answer from Gemini."}
    except Exception as e:
        app.logger.error(f"Error processing Gemini response for client {client_id}: {str(e)}")
        return {"error": f"Error processing Gemini response: {str(e)}"}

# API Routes with rate limiting
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "clients_count": len(client_data),
        "embedding_model_loaded": embedding_model is not None
    })

@app.route('/clients', methods=['GET'])
def list_clients():
    result = []
    for client_id, data in client_data.items():
        result.append({
            "client_id": client_id,
            "documents_count": len(data.loaded_documents),
            "last_active": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data.last_used_time))
        })
    return jsonify({"clients": result})

@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        # Get client ID from form data or JSON
        if request.is_json:
            client_id = request.json.get('client_id')
        else:
            client_id = request.form.get('client_id')
            
        if not client_id:
            return jsonify({"error": "client_id is required"}), 400
        
        if client_id not in client_data:
            client_data[client_id] = ClientData(client_id)
            
        client_info = client_data[client_id]
        
        if 'file' in request.files:
            file = request.files['file']
            if not file.filename:
                return jsonify({"error": "No file selected"}), 400
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            file.save(temp_file.name)
            pdf_source = temp_file.name
            document_name = file.filename
            document_url = None
        elif request.is_json and 'url' in request.json:
            pdf_source = request.json['url']
            document_name = request.json.get('document_name', pdf_source)
            
            if pdf_source in client_info.document_urls:
                return jsonify({
                    "error": "This URL has already been processed",
                    "existing_document": next(
                        (doc for doc in client_info.loaded_documents 
                         if doc.get('url') == pdf_source), None)
                }), 409
                
            document_url = pdf_source
        else:
            return jsonify({"error": "Either file upload or URL is required"}), 400
        
        text = get_pdf_content(pdf_source)
        if not text or not text.strip():
            return jsonify({"error": "No text extracted from PDF"}), 400
        
        threading.Thread(
            target=lambda: process_document(text, client_id, document_name, document_url),
            daemon=True  # Ensure threads don't block server shutdown
        ).start()
        
        return jsonify({
            "status": "processing",
            "client_id": client_id,
            "document_name": document_name,
            "document_url": document_url,
            "message": "Document upload successful. Processing started in background."
        })
    except Exception as e:
        app.logger.error(f"Upload failed: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/check-url', methods=['GET'])
def check_url():
    client_id = request.args.get('client_id')
    url = request.args.get('url')
    
    if not client_id or not url:
        return jsonify({"error": "Both client_id and url parameters are required"}), 400
    
    if client_id not in client_data:
        return jsonify({"exists": False})
    
    client_info = client_data[client_id]
    exists = url in client_info.document_urls
    document = next(
        (doc for doc in client_info.loaded_documents if doc.get('url') == url),
        None
    ) if exists else None
    
    return jsonify({
        "exists": exists,
        "document": document
    })

@app.route('/documents', methods=['GET'])
def get_documents():
    client_id = request.args.get('client_id')
    if not client_id:
        return jsonify({"error": "client_id parameter is required"}), 400
    
    if client_id not in client_data:
        return jsonify({"documents": []})
    
    return jsonify({
        "client_id": client_id,
        "documents": client_data[client_id].loaded_documents
    })

@app.route('/status', methods=['GET'])
def get_status():
    client_id = request.args.get('client_id')
    if not client_id:
        return jsonify({"error": "client_id parameter is required"}), 400
    
    if client_id not in client_data:
        return jsonify({
            "client_id": client_id,
            "exists": False,
            "documents_count": 0,
            "processing": False
        })
    
    client_info = client_data[client_id]
    return jsonify({
        "client_id": client_id,
        "exists": True,
        "documents_count": len(client_info.loaded_documents),
        "documents": client_info.loaded_documents,
        "processing": client_info.processing
    })

@app.route('/ask', methods=['POST'])
def ask():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400
        
        if 'question' not in request.json or 'client_id' not in request.json:
            return jsonify({"error": "Question and client_id are required"}), 400
        
        client_id = request.json['client_id']
        question = request.json['question']
        document_id = request.json.get('document_id')
        
        if client_id not in client_data:
            return jsonify({"error": f"Client {client_id} not found. Please upload a document first."}), 404
        
        client_info = client_data[client_id]
        if not client_info.loaded_documents:
            return jsonify({"error": "No documents have been processed for this client. Please upload a document first."}), 400
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            app.logger.error("Missing GEMINI_API_KEY in environment variables")
            return jsonify({"error": "Missing GEMINI_API_KEY in environment variables"}), 500
        
        return jsonify(ask_question(question, client_id, api_key, document_id))
    except Exception as e:
        app.logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/delete-document', methods=['DELETE'])
def delete_document():
    try:
        if not request.is_json or 'client_id' not in request.json or 'document_id' not in request.json:
            return jsonify({"error": "client_id and document_id are required"}), 400
        
        client_id = request.json['client_id']
        document_id = request.json['document_id']
        
        if client_id not in client_data:
            return jsonify({"error": f"Client {client_id} not found"}), 404
        
        client_info = client_data[client_id]
        collection = client_info.collection
        
        doc_index = None
        for i, doc in enumerate(client_info.loaded_documents):
            if doc["id"] == document_id:
                doc_index = i
                break
        
        if doc_index is None:
            return jsonify({"error": f"Document {document_id} not found for client {client_id}"}), 404
        
        removed_doc = client_info.loaded_documents[doc_index]
        if removed_doc.get('url'):
            client_info.document_urls.discard(removed_doc['url'])
        
        collection.delete(where={"document_id": document_id})
        client_info.loaded_documents.pop(doc_index)
        
        app.logger.info(f"Document {document_id} deleted for client {client_id}")
        return jsonify({
            "status": "success",
            "message": f"Document '{removed_doc['name']}' deleted successfully for client {client_id}"
        })
    except Exception as e:
        app.logger.error(f"Error deleting document: {str(e)}")
        return jsonify({"error": f"Error deleting document: {str(e)}"}), 500

@app.route('/delete-client', methods=['DELETE'])
def delete_client():
    try:
        if not request.is_json or 'client_id' not in request.json:
            return jsonify({"error": "client_id is required in JSON body"}), 400
        
        client_id = request.json['client_id']
        
        if client_id not in client_data:
            return jsonify({"error": f"Client {client_id} not found"}), 404
        
        del client_data[client_id]
        
        app.logger.info(f"Client {client_id} deleted")
        return jsonify({
            "status": "success",
            "message": f"Client {client_id} and all their documents deleted successfully"
        })
    except Exception as e:
        app.logger.error(f"Error deleting client: {str(e)}")
        return jsonify({"error": f"Error deleting client: {str(e)}"}), 500

def start_model_cleanup_thread():
    def check_model_usage():
        global embedding_model, client_data
        while True:
            time.sleep(300)  # Check every 5 minutes
            current_time = time.time()
            any_recent_activity = False
            
            for client_info in client_data.values():
                if current_time - client_info.last_used_time < 1800:  # 30 minutes
                    any_recent_activity = True
                    break
            
            with model_lock:  # Thread safety for model unloading
                if embedding_model and not any_recent_activity:
                    app.logger.info("Unloading embedding model due to inactivity")
                    embedding_model = None
    
    cleanup_thread = threading.Thread(target=check_model_usage, daemon=True)
    cleanup_thread.start()
    return cleanup_thread

# Initialize existing clients and start cleanup thread
initialize_existing_clients()
cleanup_thread = start_model_cleanup_thread()

# For production deployment with gunicorn
if __name__ != "__main__":
    # Set up logging for production
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    
    # Initialize application
    initialize_existing_clients()
    cleanup_thread = start_model_cleanup_thread()

# For development only - don't use in production
if __name__ == "__main__":
    # This block should not be used in production
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', '5000')))