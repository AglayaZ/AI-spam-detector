import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "text"]
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size = 0.2, random_state = 67)

vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectors, y_train)

prediction = model.predict(X_test_vectors)
print(f"Accuracy: {accuracy_score(y_test, prediction):.2%}")
print(classification_report(y_test, prediction))

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")