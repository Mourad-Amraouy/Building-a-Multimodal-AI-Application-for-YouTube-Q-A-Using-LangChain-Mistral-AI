import streamlit as st
import requests
import faiss
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi

# ✅ **Step 1: Get YouTube Transcript**
def get_transcript(video_url):
    try:
        video_id = video_url.split("v=")[-1]  # Extract video ID
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = "\n".join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        return f"Error retrieving transcript: {e}"

# ✅ **Step 2: Generate Embeddings**
def create_vector_index(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embedding_model = "BAAI/bge-small-en"  # Optimized for retrieval
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    embedding_vectors = [embeddings.embed_query(chunk) for chunk in chunks]
    
    # Create FAISS index
    dimension = len(embedding_vectors[0])
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embedding_vectors, dtype="float32"))
    
    return chunks, embeddings, faiss_index

# ✅ **Step 3: Ask Mistral AI**
MISTRAL_API_KEY = "sk-your-mistral-api-key"  # Replace with a valid API key

def ask_mistral(question, faiss_index, chunks, embeddings):
    API_URL = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}

    # Retrieve most relevant transcript chunks
    D, I = faiss_index.search(np.array([embeddings.embed_query(question)], dtype="float32"), k=3)
    context = "\n".join([chunks[i] for i in I[0]])

    payload = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": "You are an AI assistant answering questions based on a video transcript."},
            {"role": "user", "content": f"Based on this transcript:\n{context}\n\nAnswer this: {question}"}
        ],
        "temperature": 0.7
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.text}"

# ✅ **Step 4: Build Streamlit Web App**
st.title("📺 YouTube Q&A Bot with Mistral AI")
st.write("Enter a YouTube video URL, then ask a question about its content.")

video_url = st.text_input("🔗 Enter YouTube Video URL:")
question = st.text_input("❓ Ask a question about the video:")

if st.button("Generate Answer"):
    if video_url:
        st.write("📥 Retrieving transcript...")
        transcript_text = get_transcript(video_url)

        if "Error" not in transcript_text:
            st.write("📊 Processing transcript embeddings...")
            chunks, embeddings, faiss_index = create_vector_index(transcript_text)

            st.write("🤖 Generating answer from Mistral AI...")
            answer = ask_mistral(question, faiss_index, chunks, embeddings)

            st.success("✅ Answer:")
            st.write(answer)
        else:
            st.error("🚨 Unable to retrieve transcript. Try another video.")
    else:
        st.warning("⚠️ Please enter a valid YouTube video URL.")

