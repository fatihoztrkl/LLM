import hashlib
import json
import os
import re
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

turkish_stopwords = set([
    "nedir", "ne", "demek", "açıkla", "örnek", "ver", "ile", "ilgili", "açıklayınız",
    "açıklayın", "kısaca", "tanımla", "tanımı", "anlat", "nedeni", "amaç", "nasıl", "neden","verir"
])

def clean_question(text):
    cleaned = clean_text(text)
    words = cleaned.split()
    return " ".join([w for w in words if w not in turkish_stopwords]).strip()

with open("hocaAnahtar.json", "r", encoding="utf-8") as f:
    ml_keywords_raw = json.load(f)

ml_keywords = [clean_text(k) for k in ml_keywords_raw]

embedding_cache_file = "ml_keyword_embeddings.json"
hash_cache_file = "ml_keywords_hash.txt"

def compute_hash(lst):
    data = json.dumps(lst, ensure_ascii=False).encode('utf-8')
    return hashlib.md5(data).hexdigest()

current_hash = compute_hash(ml_keywords)

if os.path.exists(hash_cache_file):
    with open(hash_cache_file, "r", encoding="utf-8") as f:
        saved_hash = f.read().strip()
else:
    saved_hash = None

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

def is_ml_related(question, threshold=0.7):
    cleaned = clean_question(question)
    if not cleaned:
        print("❌ Soru yalnızca genel ifadeler içeriyor (örnek: 'nedir').")
        return False
    question_embedding = model.encode([cleaned])[0]
    similarities = cosine_similarity([question_embedding], keyword_embeddings)[0]
    max_sim = np.max(similarities)
    matched_keyword = ml_keywords[np.argmax(similarities)]

    if max_sim >= threshold:
        print(f"✅ ML ile ilgili (benzerlik: {max_sim:.2f}) — eşleşen anahtar kelime: '{matched_keyword}'")
        return True
    else:
        print(f"❌ ML ile ilgili değil (benzerlik: {max_sim:.2f})")
        return False

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

client = chromadb.Client()

try:
    collection = client.create_collection("machine_learning_faq")
except Exception:
    collection = client.get_collection("machine_learning_faq")

with open("machine_learning_faq.json", "r", encoding="utf-8") as f:
    df = json.load(f)
entries = df["data"]

questions = [item['question'] for item in entries]
embeddings = model.encode([clean_question(q) for q in questions])

for i, embedding in enumerate(embeddings):
    collection.add(
        documents=[entries[i]["answer"]],
        metadatas=[{"question": entries[i]["question"]}],
        ids=[str(entries[i]["id"])],
        embeddings=[embedding.tolist()]
    )

def ask_satisfaction(first_time=True):
    while True:
        if first_time:
            satis = input("\nCevaptan memnun musunuz? (E/H): ").strip().lower()
            if satis in ["e", "h"]:
                return satis
            else:
                print("Lütfen sadece 'E' veya 'H' giriniz.")
        else:
            satis = input("\nAlternatif cevaptan memnun musunuz? (Sadece 'E' girin): ").strip().lower()
            if satis == "e":
                return satis
            else:
                print("Lütfen sadece 'E' giriniz.")

while True:
    user_question = input("\nLütfen sorunuzu giriniz (Çıkmak için 'q' basın): ")
    if user_question.strip().lower() == 'q':
        print("Program sonlandırıldı.")
        break

    cleaned_user_question = clean_question(user_question)

    if not is_ml_related(user_question):
        print("Bu soru makine öğrenmesi ile ilgili değil. Yanıt verilmiyor. Size sadece alanınız içinde cevaplar verebilirim.")
        continue

    question_vector = model.encode([cleaned_user_question])[0]

    results = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=1,
        include=["embeddings", "documents", "metadatas"]
    )

    answer_from = None
    answer_text = ""

    if results['documents'] and results['embeddings'] and len(results['embeddings'][0]) > 0:
        top_embedding = np.array(results['embeddings'][0][0])
        similarity = cosine_similarity([question_vector], [top_embedding])[0][0]

        threshold = 0.7
        if similarity >= threshold:
            answer_from = "db"
            answer_text = results['documents'][0][0]
            print(f"Cevap bulundu (Benzerlik: {similarity:.2f}):")
            print(answer_text)
        else:
            answer_from = "api"
    else:
        answer_from = "api"

    if answer_from == "api":
        print("API'ye yönlendiriliyor...")
        answer_text = "Bu soruya verilecek örnek bir API cevabıdır."
        print("API Cevabı:", answer_text)

    satis = ask_satisfaction(first_time=True)

    while satis == "h":
        if answer_from == "db":
            print("API'ye yönlendiriliyor alternatif cevap için...")
            alt_answer = "Alternatif API cevabı burada."
            print("Alternatif API Cevabı:", alt_answer)
            answer_from = "api"
            answer_text = alt_answer
        else:
            print("API'ye tekrar yönlendiriliyor alternatif cevap için...")
            alt_answer = "API'den ikinci alternatif cevap."
            print("Alternatif API Cevabı:", alt_answer)
            answer_text = alt_answer

        satis = ask_satisfaction(first_time=False)
