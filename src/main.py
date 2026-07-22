import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load datasets
fake = pd.read_csv("src/data/Fake.csv")
true = pd.read_csv("src/data/True.csv")

fake['label'] = 0
true['label'] = 1

data = pd.concat([fake, true], axis=0)[['text', 'label']]

X = data['text']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

# Save model + vectorizer into backend folder
pickle.dump(model, open("backend/model.pkl", "wb"))
pickle.dump(vectorizer, open("backend/vectorizer.pkl", "wb"))
