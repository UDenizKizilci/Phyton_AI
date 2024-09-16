import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews

# 1. Adım: Gerekli Kütüphaneleri Eklemek

# 2. Adım: Veri Hazırlığı
nltk.download("movie_reviews")

# Veri Yüklemesi
documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# DataFrame'e Dönüştürme
df = pd.DataFrame(documents, columns=["review", "sentiment"])

# Düşünce Etiketleme
df["sentiment"] = df["sentiment"].map({"pos": 1, "neg": 0})

# 3. Adım: Model Train Etme

# Metin Verilerini Vektörlere Dönüştürme
vectorizer = CountVectorizer(max_features=2000, stop_words='english')
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Veriyi Train ve Test'lere Ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Naive Bayes Sınıflandırıcısı
model = MultinomialNB()
model.fit(X_train, y_train)

# Model Değerlendirmesi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# 4.Adım : Tahmin Etme

def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return "Positive" if prediction[0] == 1 else "Negative"

# Tahmin Fonksiyonunu Test Etme
test_reviews = [
    "I absolutely loved this movie! It was fantastic.",
    "It was a terrible film. I hated it.",
    "The movie was okay, nothing special."
]

for review in test_reviews:
    print(f"Review: {review}\nPredicted Sentiment: {predict_sentiment(review)}\n")
