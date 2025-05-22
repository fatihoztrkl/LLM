from thefuzz import process
import hashlib
import json
import os
import re
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Model yükle
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Temizlik fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

turkish_stopwords = set([
    "nedir", "ne", "demek", "açıkla", "örnek", "ver", "ile", "ilgili", "açıklayınız",
    "açıklayın", "kısaca", "tanımla", "tanımı", "anlat", "nedeni", "amaç", "nasıl", "neden"
])

def clean_question(text):
    cleaned = clean_text(text)
    words = cleaned.split()
    return " ".join([w for w in words if w not in turkish_stopwords]).strip()

# Hoca anahtar terimleri dosyasından oku
with open("hocaAnahtar.json", "r", encoding="utf-8") as f:
    known_terms_raw = json.load(f)

# Temizlenmiş terimler
known_terms = [clean_text(term) for term in known_terms_raw]

# Yazım düzeltme fonksiyonu (thefuzz ile)
def find_best_match(user_input, terms, threshold=80):
    best_match = process.extractOne(user_input, terms)
    if best_match and best_match[1] >= threshold:
        return best_match[0]
    return None

# ML anahtar kelimeler (aynı hocaAnahtar.json dosyasını kullanabiliriz)
ml_keywords = known_terms

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

# ChromaDB Client
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

# Kullanıcıdan soru al
user_question = input("Lütfen sorunuzu giriniz: ")

# 1. Yazım düzeltme: Kullanıcının sorusunu known_terms ile karşılaştır ve en iyi eşleşmeyi bul
cleaned_input = clean_text(user_question)
corrected_term = find_best_match(cleaned_input, known_terms, threshold=80)

if corrected_term:
    print(f"Yazım düzeltme önerisi: '{corrected_term}' (Orijinal: '{user_question}')")
    question_for_search = corrected_term
else:
    question_for_search = user_question

# 2. ML ile ilgili mi kontrol et
if not is_ml_related(question_for_search):
    print("Bu soru makine öğrenmesi ile ilgili değil. Yanıt verilmiyor.")
else:
    cleaned_question = clean_question(question_for_search)
    question_vector = model.encode([cleaned_question])[0]

    results = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=1,
        include=["embeddings", "documents", "metadatas"]
    )

    if results['documents'] and results['embeddings'] and len(results['embeddings'][0]) > 0:
        top_embedding = np.array(results['embeddings'][0][0])
        similarity = cosine_similarity([question_vector], [top_embedding])[0][0]

        threshold = 0.75
        if similarity >= threshold:
            print(f"Cevap bulundu (Benzerlik: {similarity:.2f}):")
            print(results['documents'][0][0])
        else:
            print("Veritabanında uygun cevap bulunamadı, API'ye yönlendiriliyor...")

            # --- API çağrısı buraya yapılacak ---
            # Örnek: OpenAI API ile çağrı yapıp cevabı al
            # dummy_api_response yerine API'den gelen gerçek yanıt atanacak
            dummy_api_response = "Bu soruya verilecek örnek API cevabıdır."

            print("API Cevabı:", dummy_api_response)
            save_to_new_file(user_question, dummy_api_response)
    else:
        print("Veritabanında uygun cevap bulunamadı, API'ye yönlendiriliyor...")

        # --- API çağrısı buraya yapılacak ---
        dummy_api_response = "Bu soruya verilecek örnek API cevabıdır."

        print("API Cevabı:", dummy_api_response)
        save_to_new_file(user_question, dummy_api_response)
