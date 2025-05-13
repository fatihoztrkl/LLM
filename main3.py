import json
import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Anahtar kelimeleri yükle
with open("ml_keywords.json", "r", encoding="utf-8") as f:
    ml_keywords = json.load(f)

# Anahtar kelimelerle kontrol fonksiyonu
def is_ml_related(question):
    question = question.lower()
    for keyword in ml_keywords:
        if keyword.lower() in question:
            return True
    return False

# Yeni verileri ayrı dosyada kaydet
def save_to_new_file(question, answer, filename="new_ml_data.json"):
    new_entry = {"question": question, "answer": answer}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"data": []}
    data["data"].append(new_entry)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"\n[Soru ve cevap '{filename}' dosyasına kaydedildi.]")

# ChromaDB Client Başlatma
client = chromadb.Client()

# Koleksiyon oluşturma
collection = client.create_collection("machine_learning_faq")

# Cümle gömme modeli
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

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

# Kullanıcı sorusunu al
user_question = input("Lütfen sorunuzu giriniz: ")

# 1. Makine öğrenmesiyle ilgili mi?
if not is_ml_related(user_question):
    print("Bu soru makine öğrenmesi ile ilgili değil. Yanıt verilmiyor.")
else:
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
            print(f"Cevap bulundu (Benzerlik: {similarity:.2f}):")
            print(results['documents'][0][0])
        else:
            print(f"Benzerlik çok düşük ({similarity:.2f}). API'ye yönlendiriliyor...")
            dummy_api_response = "Bu soruya verilecek örnek bir API cevabıdır."  # Örnek cevap
            print("API Cevabı:", dummy_api_response)
            save_to_new_file(user_question, dummy_api_response)
    else:
        print("Veritabanında benzer bir cevap bulunamadı. API'ye yönlendiriliyor...")
        dummy_api_response = "Bu soruya verilecek örnek bir API cevabıdır."  # Örnek cevap
        print("API Cevabı:", dummy_api_response)
        save_to_new_file(user_question, dummy_api_response)
