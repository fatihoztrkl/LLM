from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import hashlib
import logging
import openai
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

# ChromaDB Client başlatma
client = chromadb.Client()
collection = client.get_or_create_collection("machine_learning_faq")

# Cümle gömme modeli
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Anahtar kelimeleri yükle
try:
    with open("ml_keywords.json", "r", encoding="utf-8") as f:
        ml_keywords = json.load(f)
except FileNotFoundError:
    logging.error("Anahtar kelime dosyası bulunamadı!")
    ml_keywords = []

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
    logging.debug(f"Benzerlik: {max_sim:.2f}, Eşleşen kelime: {matched_keyword}")
    return max_sim >= threshold

# Yeni verileri ayrı dosyada kaydet
def save_to_new_file(question, answer, filename="new_ml_data.json"):
    new_entry = {"question": question, "answer": answer}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"data": []}

    if new_entry not in data["data"]:
        data["data"].append(new_entry)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"[Yeni soru '{filename}' dosyasına kaydedildi.]")
    else:
        logging.info(f"[Soru zaten mevcut, '{filename}' dosyasına tekrar eklenmedi.]")

from openai import OpenAI

client = OpenAI()

def query_openai_and_save(question):
    logging.info("OpenAI API çağrısı yapılıyor...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Kullanıcının makine öğrenmesiyle ilgili sorusuna kısa ve açık şekilde cevap ver."},
            {"role": "user", "content": question}
        ],
        max_tokens=0
    )
    answer = response.choices[0].message.content
    save_to_new_file(question, answer)
    return answer



# Soru-cevap verisini yükleme
try:
    with open("machine_learning_faq.json", "r", encoding="utf-8") as f:
        df = json.load(f)
    entries = df["data"]
except FileNotFoundError:
    logging.error("Soru-cevap verisi bulunamadı!")
    entries = []

# Koleksiyon boşsa verileri yükle
if collection.count() == 0 and entries:
    questions = [item['question'] for item in entries]
    embeddings = model.encode(questions)

    for i, embedding in enumerate(embeddings):
        collection.add(
            documents=[entries[i]["answer"]],
            metadatas=[{"question": entries[i]["question"]}],
            ids=[str(entries[i]["id"])],
            embeddings=[embedding.tolist()]
        )
    logging.info("Veritabanına veriler eklendi.")
else:
    logging.info("Veriler zaten veritabanında mevcut.")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    user_question = request.question
    logging.debug(f"User question: {user_question}")

    if not is_ml_related(user_question):
        return {"answer": "Bu soru makine öğrenmesi ile ilgili değil. Yanıt verilmiyor."}

    question_vector = model.encode([user_question])[0]

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
            return {
                "answer": results['documents'][0][0],
                "similarity": similarity,
                "source_question": results['metadatas'][0][0]['question']
            }
        else:
            answer = query_openai_and_save(user_question)
            logging.debug(f"Answer: {answer}")

            return {
                "answer": answer,
                "similarity": similarity
            }
    else:
        answer = query_openai_and_save(user_question)
        logging.debug(f"Answer: {answer}")
        return {
            "answer": answer,
            "similarity": None
        }


