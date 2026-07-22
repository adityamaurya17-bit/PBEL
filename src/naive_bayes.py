import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv("src/data/Fake.csv")
true = pd.read_csv("src/data/True.csv")

# Add labels
fake['label'] = 0
true['label'] = 1

# Combine
data = pd.concat([fake, true], axis=0)
data = data[['text', 'label']]

X = data['text']
y = data['label']

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
