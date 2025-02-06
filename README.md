# Building a Multimodal AI Application for YouTube Q&A Using LangChain & Mistral AI

## 📌 Project Overview
This project implements a YouTube Q&A bot that:
- Extracts video transcripts
- Generates embeddings
- Stores them in a FAISS vector database
- Answers user queries using the Mistral AI API
- Deploys as a web application using Gradio

---

## 📝 Introduction
With the rise of online education and digital content, extracting meaningful insights from YouTube videos has become essential. This project leverages artificial intelligence (AI) to build a question-answering bot capable of retrieving information from YouTube transcripts and answering user queries efficiently.

---

## 🔍 Methodology
The project follows these key steps:
1. Extract the video transcript using `youtube-transcript-api`.
2. Split the text into smaller chunks for efficient retrieval.
3. Generate embeddings using `HuggingFaceEmbeddings`.
4. Store embeddings in a FAISS vector database.
5. Retrieve relevant transcript chunks based on user queries.
6. Use the Mistral AI API to generate responses.

---

## 🛠️ Installation & Dependencies

To run the project, install the required Python packages:
```bash
pip install gradio yt-dlp youtube-transcript-api langchain faiss-cpu \
            sentence-transformers requests
```

---

## 🏗️ Implementation

### 1️⃣ Extract YouTube Transcript
```python
from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = "\n".join([entry['text'] for entry in transcript])
    return text
```

### 2️⃣ Generate Embeddings & Store in FAISS
```python
import faiss
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_index(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embedding_model = "BAAI/bge-small-en"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    embedding_vectors = [embeddings.embed_query(chunk) for chunk in chunks]
    dimension = len(embedding_vectors[0])
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embedding_vectors, dtype="float32"))
    return chunks, embeddings, faiss_index
```

### 3️⃣ Ask Mistral AI for Answers
```python
import requests

MISTRAL_API_KEY = "sk-your-mistral-api-key"

def ask_mistral(video_url, question):
    transcript_text = get_transcript(video_url)
    chunks, embeddings, faiss_index = create_vector_index(transcript_text)
    API_URL = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    D, I = faiss_index.search(np.array([embeddings.embed_query(question)], dtype="float32"), k=3)
    context = "\n".join([chunks[i] for i in I[0]])
    payload = {"model": "mistral-tiny", "messages": [{"role": "user", "content": f"{question}"}], "temperature": 0.7}
    response = requests.post(API_URL, json=payload, headers=headers)
    return response.json()["choices"][0]["message"]["content"]
```

### 4️⃣ Deploy as a Web App Using Gradio
```python
import gradio as gr

app = gr.Interface(
    fn=ask_mistral,
    inputs=[gr.Textbox(label="Enter YouTube Video URL"), gr.Textbox(label="Ask a question about the video")],
    outputs="text",
    title="YouTube Q&A Bot with Mistral AI",
)
app.launch()
```

---

## ✅ Results
The system successfully extracts YouTube video transcripts and provides relevant answers based on user queries. The use of FAISS for similarity search improves retrieval accuracy.

---

## 📌 Conclusion
This project demonstrates the effectiveness of AI in processing and understanding YouTube video content. Future work could include:
- Multi-language support
- Improved retrieval techniques
- Integration with other AI models

---

## 📚 References
- [LangChain Documentation](https://python.langchain.com/)
- [Mistral AI](https://mistral.ai/)
- [FAISS](https://faiss.ai/)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)

---

🚀 **Built by Mourad Amraouy**  
📧 Contact: [moradamraouy@gmail.com](mailto:moradamraouy@gmail.com)
