# 🧠 LLM Destekli Soru-Cevap Uygulaması

Bu proje, kullanıcıların makine öğrenmesiyle ilgili sorularını yanıtlayan bir web API sunar. Uygulama, öncelikle ChromaDB vektör veritabanında benzer soruları arar. Yeterli benzerlik bulunamazsa, OpenAI'nin GPT-3.5-Turbo modeli kullanılarak yanıt oluşturulur. Yeni sorular ve yanıtlar, gelecekteki sorgular için veritabanına eklenir.

## 🚀 Özellikler

- **Benzerlik Tabanlı Arama:** ChromaDB kullanarak önceki sorular arasında benzerlik arar.
- **OpenAI Entegrasyonu:** Yeterli benzerlik bulunamazsa, GPT-3.5-Turbo modeli ile yanıt oluşturur.
- **Veritabanı Güncelleme:** Yeni sorular ve yanıtlar otomatik olarak veritabanına eklenir.
- **FastAPI ile Web API:** Hızlı ve modern bir web API sunar.
- **Test Modu:** Geliştirme ve test süreçleri için özel mod.

## 📦 Kurulum

1. **Depoyu Klonlayın:**

   ```bash
   git clone https://github.com/halilkorkudan/LLM.git
   cd LLM
````

2. **Sanal Ortam Oluşturun (Opsiyonel):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows için: venv\Scripts\activate
   ```

3. **Gereksinimleri Yükleyin:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ortam Değişkenlerini Ayarlayın:**

   `.env` dosyası oluşturarak OpenAI API anahtarınızı ekleyin:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

## 🧪 Kullanım

Uygulamayı başlatmak için:

```bash
uvicorn api:app --reload
```

API'yi test etmek için, aşağıdaki gibi bir POST isteği gönderebilirsiniz:

```bash
curl -X POST http://localhost:8000/ask/ \
     -H "Content-Type: application/json" \
     -d '{"question": "Makine öğrenmesi nedir?"}'
```

