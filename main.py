import hashlib
import json
import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Anahtar kelimeleri yükle
with open("ml_keywords.json", "r", encoding="utf-8") as f:
    ml_keywords = json.load(f)

embedding_cache_file = "ml_keyword_embeddings.json"
hash_cache_file = "ml_keywords_hash.txt"

# Anahtar kelime listesinin hash'ini hesapla
def compute_hash(lst):
    data = json.dumps(lst, ensure_ascii=False).encode('utf-8')
    return hashlib.md5(data).hexdigest()

current_hash = compute_hash(ml_keywords)

# Hash dosyası varsa oku
if os.path.exists(hash_cache_file):
    with open(hash_cache_file, "r", encoding="utf-8") as f:
        saved_hash = f.read().strip()
else:
    saved_hash = None

# Hash'ler uyuşuyorsa önbelleği kullan, değilse tekrar hesapla
if saved_hash == current_hash and os.path.exists(embedding_cache_file):
    with open(embedding_cache_file, "r", encoding="utf-8") as f:
        keyword_embeddings_list = json.load(f)
    keyword_embeddings = np.array(keyword_embeddings_list)
else:
    keyword_embeddings = model.encode(ml_keywords)
    with open(embedding_cache_file, "w", encoding="utf-8") as f:
        json.dump(keyword_embeddings.tolist(), f, ensure_ascii=False, indent=4)
    with open(hash_cache_file, "w", encoding="utf-8") as f:
        f.write(current_hash)

# Embedding tabanlı ML kontrol fonksiyonu
def is_ml_related(question, threshold=0.6):
    question_embedding = model.encode([question])[0]
    similarities = cosine_similarity([question_embedding], keyword_embeddings)[0]
    max_sim = np.max(similarities)
    matched_keyword = ml_keywords[np.argmax(similarities)]

    if max_sim >= threshold:
        print(f"✅ ML ile ilgili (benzerlik: {max_sim:.2f}) — eşleşen anahtar kelime: '{matched_keyword}'")
        return True
    else:
        print(f"❌ ML ile ilgili değil (benzerlik: {max_sim:.2f})")
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
try:
    collection = client.create_collection("machine_learning_faq")
except Exception:
    collection = client.get_collection("machine_learning_faq")

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

# Kullanıcıdan soru al
user_question = input("Lütfen sorunuzu giriniz: ")

# 1. Makine öğrenmesi ile ilgili mi?
if not is_ml_related(user_question):
    print("Bu soru makine öğrenmesi ile ilgili değil. Yanıt verilmiyor. Size sadece alanınız içinde cevaplar verebilirim.")
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
            print(f"API'ye yönlendiriliyor...")
            dummy_api_response = "Bu soruya verilecek örnek bir API cevabıdır."  
            print("API Cevabı:", dummy_api_response)
            save_to_new_file(user_question, dummy_api_response)
    else:
        print("Veritabanında benzer bir cevap bulunamadı. API'ye yönlendiriliyor...")
        dummy_api_response = "Bu soruya verilecek örnek bir API cevabıdır."  
        print("API Cevabı:", dummy_api_response)
        save_to_new_file(user_question, dummy_api_response)
