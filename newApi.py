import hashlib
import json
import os
import re
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from openai import OpenAI  # Yeni OpenAI istemcisi
from dotenv import load_dotenv
load_dotenv()


# --- OpenAI istemcisi ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Model ve Temizlik Fonksiyonları ---

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

turkish_stopwords = set([
    "nedir", "ne", "demek", "açıkla", "örnek", "ver", "ile", "ilgili", "açıklayınız",
    "açıklayın", "kısaca", "tanımla", "tanımı", "anlat", "nedeni", "amaç", "nasıl", "neden", "verir"
])

def clean_question(text: str) -> str:
    cleaned = clean_text(text)
    words = cleaned.split()
    return " ".join([w for w in words if w not in turkish_stopwords]).strip()

# --- Anahtar Kelimeler ve Embedding Cache ---

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

def is_ml_related(question: str, threshold=0.7) -> bool:
    cleaned = clean_question(question)
    if not cleaned:
        return False
    question_embedding = model.encode([cleaned])[0]
    similarities = cosine_similarity([question_embedding], keyword_embeddings)[0]
    max_sim = np.max(similarities)
    return max_sim >= threshold

# --- ChromaDB ve Soru-Cevap Verisi ---

client_chroma = chromadb.Client()

try:
    collection = client_chroma.create_collection("machine_learning_faq")
except Exception:
    collection = client_chroma.get_collection("machine_learning_faq")

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

# --- OpenAI API Çağrısı ---

def ask_openai_api(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sen makine öğrenmesi alanında uzman bir asistansın. Sadece makine öğrenmesiyle ilgili sorulara kısa ve öz cevap ver; eğer soru bu alanla ilgili değilse, vereceğin cevap 'Bu soru makine öğrenmesi ile ilgili değil. Yanıt verilemiyor.' olsun."},
                {"role": "user", "content": question}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API cevabı alınamadı: {e}"

# --- Veri Kaydetme ---

def save_to_new_file(question: str, answer: str, filename="new_ml_data.json"):
    new_entry = {"question": question, "answer": answer}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"data": []}
    data["data"].append(new_entry)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- API Modelleri ---

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    from_source: str  # "db" veya "api"

class QuizRequest(BaseModel):
    questions: list[str]

class QuizResponse(BaseModel):
    quiz_questions: list[str]

class FeedbackRequest(BaseModel):
    approved: bool
    question: str
    last_answer: str

# --- FastAPI Uygulaması ---

app = FastAPI()
asked_questions_global = []

@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    user_question = req.question.strip()
    if not user_question:
        return AnswerResponse(answer="Soru boş olamaz.", from_source="none")

    asked_questions_global.append(user_question)

    if not is_ml_related(user_question):
        return AnswerResponse(answer="Bu soru makine öğrenmesi ile ilgili değil. Yanıt verilemiyor.", from_source="none")

    cleaned_user_question = clean_question(user_question)
    question_vector = model.encode([cleaned_user_question])[0]

    results = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=1,
        include=["embeddings", "documents", "metadatas"]
    )

    answer_from = "api"
    answer_text = ""

    if results['documents'] and results['embeddings'] and len(results['embeddings'][0]) > 0:
        top_embedding = np.array(results['embeddings'][0][0])
        similarity = cosine_similarity([question_vector], [top_embedding])[0][0]

        if similarity >= 0.7:
            answer_from = "db"
            answer_text = results['documents'][0][0]

    if answer_from == "api":
        answer_text = ask_openai_api(user_question)

    return AnswerResponse(answer=answer_text, from_source=answer_from)

@app.post("/quiz", response_model=QuizResponse)
def start_quiz(req: QuizRequest):
    return QuizResponse(quiz_questions=req.questions)

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.approved:
        return {"message": "Teşekkürler, cevaptan memnun kaldığınız için sevindik."}
    else:
        alt_answer = ask_openai_api(req.question + "Sen makine öğrenmesi alanında uzman bir asistansın. Kullanıcının sorusuna en iyi cevabı kısa ve öz bir şekilde cevapla.")
        return {"answer": alt_answer}

@app.get("/asked-questions")
def get_asked_questions():
    return {"asked_questions": asked_questions_global}
