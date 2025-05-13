from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# FastAPI uygulaması
app = FastAPI()

# ChromaDB Client Başlatma
client = chromadb.Client()
collection = client.create_collection("machine_learning_faq")

# Cümle gömme modeli
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Anahtar kelimeleri yükle
with open("ml_keywords.json", "r", encoding="utf-8") as f:
    ml_keywords = json.load(f)

# Soru-cevap verisini yükleme
with open("machine_learning_faq.json", "r", encoding="utf-8") as f:
    df = json.load(f)
entries = df["data"]

# Soruları vektörleştir
questions = [item['question'] for item in entries]
embeddings = model.encode(questions)

# Veritabanına ekle
for i, embedding in enumerate(embeddings):
    collection.add(
        documents=[entries[i]["answer"]],
        metadatas=[{"question": entries[i]["question"]}],
        ids=[str(entries[i]["id"])],
        embeddings=[embedding.tolist()]
    )

# Anahtar kelimelerle kontrol fonksiyonu
def is_ml_related(question):
    question = question.lower()
    for keyword in ml_keywords:
        if keyword.lower() in question:
            return True
    return False

# Kullanıcı sorusu için API endpoint
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    user_question = request.question
    
    # 1. Makine öğrenmesiyle ilgili mi?
    if not is_ml_related(user_question):
        return {"message": "Bu soru makine öğrenmesi ile ilgili değil. Yanıt verilmiyor."}
    
    # 2. Soruyu vektörleştir
    question_vector = model.encode([user_question])[0]

    # 3. Veritabanında benzerini ara
    results = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=1,
        include=["embeddings", "documents", "metadatas"]
    )

    # 4. Sonuç kontrolü ve benzerlik eşiği uygulama
    if results['documents'] and results['embeddings'] and len(results['embeddings'][0]) > 0:
        top_embedding = np.array(results['embeddings'][0][0])
        similarity = cosine_similarity([question_vector], [top_embedding])[0][0]

        threshold = 0.75  # benzerlik eşiği
        if similarity >= threshold:
            return {"answer": results['documents'][0][0], "similarity": similarity}
        else:
            # Benzerlik düşükse, API'ye yönlendiriliyor
            dummy_api_response = "Bu soruya verilecek örnek bir API cevabıdır."
            save_to_new_file(user_question, dummy_api_response)  # Yeni veriyi kaydedelim
            return {"message": f"Benzerlik çok düşük ({similarity:.2f}). API'ye yönlendiriliyor..."}
    else:
        # Veritabanında cevap yoksa, API'ye yönlendiriliyor
        dummy_api_response = "Bu soruya verilecek örnek bir API cevabıdır."
        save_to_new_file(user_question, dummy_api_response)  # Yeni veriyi kaydedelim
        return {"message": "Veritabanında benzer bir cevap bulunamadı. API'ye yönlendiriliyor..."}


