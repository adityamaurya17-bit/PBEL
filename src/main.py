import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def strip_source_artifacts(text):
    # True.csv is ~99% Reuters wire copy with a leading "CITY (Reuters) - "
    # dateline; Fake.csv almost never has one. Left in, the model just learns
    # to detect Reuters formatting instead of anything about truthfulness.
    text = re.sub(r"^.{0,120}?\(Reuters\)\s*-\s*", "", text, count=1)
    text = re.sub(r"\bReuters\b", "", text, flags=re.IGNORECASE)
    return text


# ISOT: full-length political news articles (2016-17), Fake vs Reuters-sourced True
fake = pd.read_csv("src/data/Fake.csv")
true = pd.read_csv("src/data/True.csv")
fake["label"] = 0
true["label"] = 1
isot = pd.concat([fake, true], axis=0)[["text", "label"]]
isot["text"] = isot["text"].apply(strip_source_artifacts)

# LIAR: short PolitiFact-checked political statements across many subjects
# (economy, healthcare, immigration, etc.), not just 2016-17 US news.
liar_cols = [
    "id", "label", "statement", "subject", "speaker", "job", "state",
    "party", "barely_true_c", "false_c", "half_true_c", "mostly_true_c",
    "pants_fire_c", "context",
]
liar = pd.concat([
    pd.read_csv("src/data/external/liar_train.tsv", sep="\t", names=liar_cols),
    pd.read_csv("src/data/external/liar_valid.tsv", sep="\t", names=liar_cols),
    pd.read_csv("src/data/external/liar_test.tsv", sep="\t", names=liar_cols),
], axis=0)
liar_true = {"true", "mostly-true", "half-true"}
liar_false = {"false", "barely-true", "pants-fire"}
liar = liar[liar["label"].isin(liar_true | liar_false)]
liar["label"] = liar["label"].isin(liar_true).astype(int)
liar = liar[["statement", "label"]].rename(columns={"statement": "text"})

# FEVER: general-knowledge Wikipedia claims (any topic, not just politics).
# REFUTES claims are minimally-edited mutations of SUPPORTS claims, which
# forces the model to key off actual content instead of writing style.
fever = pd.read_csv("src/data/external/fever_claims.csv")

data = pd.concat([isot, liar, fever], axis=0).dropna(subset=["text"])
data = data[data["text"].str.strip() != ""]

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

# Save model + vectorizer into backend folder
pickle.dump(model, open("backend/model.pkl", "wb"))
pickle.dump(vectorizer, open("backend/vectorizer.pkl", "wb"))
