import json

# Makine öğrenmesi ile ilgili anahtar kelimeler listesi
ml_keywords = [
    "makine öğrenmesi", "machine learning", "yapay zeka", "artificial intelligence", "ai", "ml", "veri", "data",
    "model", "eğitim", "test", "validation", "performans", "özellik", "feature", "label", "etiket", "predict", "tahmin",
    "sınıflandırma", "classification", "regresyon", "regression", "kümeleme", "clustering",
    "k-en yakın komşu", "k-nn", "destek vektör makineleri", "svm", "naive bayes", "karar ağacı", "random forest",
    "xgboost", "lightgbm", "gradient boosting", "lojistik regresyon", "logistic regression", "doğrusal regresyon",
    "derin öğrenme", "deep learning", "nöral ağ", "neural network", "cnn", "rnn", "transformer", "lstm",
    "gpt", "bert", "autoencoder", "attention", "resnet", "mobilenet",
    "doğruluk", "accuracy", "hata", "loss", "precision", "recall", "f1", "auc", "roc", "mse", "rmse",
    "denetimli", "supervised", "denetimsiz", "unsupervised", "takviyeli öğrenme", "reinforcement learning",
    "yarı denetimli", "semi-supervised",
    "veri temizleme", "ön işleme", "feature engineering", "veri seti", "dataset", "boyut indirgeme", "dimensionality reduction",
    "pca", "normalizasyon", "ölçekleme", "standardizasyon", "grid search", "cross validation", "hyperparameter tuning",
    "sklearn", "scikit-learn", "tensorflow", "keras", "pytorch", "huggingface", "mlflow", "onnx",
    "bias", "variance", "overfitting", "underfitting", "generalization", "embedding", "feature importance",
    "explainability", "shap", "lime", "model yorumlanabilirliği"
]

# JSON dosyasına kaydetme
with open("ml_keywords.json", "w", encoding="utf-8") as f:
    json.dump(ml_keywords, f, ensure_ascii=False, indent=4)

print("Anahtar kelimeler 'ml_keywords.json' dosyasına kaydedildi.")
