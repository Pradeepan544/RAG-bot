from flask import Flask, request, jsonify, render_template, send_from_directory
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing! Add it to the .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_collection(name="3d_printing_knowledge")

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

app = Flask(__name__)

# Function to generate embeddings
def generate_embedding(text):
    return embedder.encode(text).tolist()

# Function to retrieve relevant context from ChromaDB
def search_query(query_text, top_k=3):
    query_embedding = generate_embedding(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k
    )
    print(results['metadatas'])
    # Extract relevant contexts
    documents = results.get("documents", [[]])[0]
    metadata = results.get("metadatas", [[]])[0]
    
    # Extract image paths and convert to relative URLs
    image_urls = []
    for meta in metadata:
        image_paths = meta.get("image_paths", "")
        if image_paths:
            # Convert absolute paths to /static/images URLs
            image_list = image_paths.split(",")
            image_urls.extend([f"/static/images/{os.path.basename(img)}" for img in image_list])
    
    return documents, image_urls

# Function to get response from Gemini
def ask_gemini(query, context):
    prompt = f"Context: {context}\n\nUser Query: {query}\nAnswer: , if no answer is found provide a relevant answer for it"
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text

# API Route for Chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")

    # Retrieve relevant contexts from ChromaDB
    relevant_contexts, image_urls = search_query(user_query)

    # Get response from Gemini
    response = ask_gemini(user_query, " ".join(relevant_contexts))
    
    return jsonify({"response": response, "images": image_urls})

# Route to serve images from /static/images/
@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

# Frontend route
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)