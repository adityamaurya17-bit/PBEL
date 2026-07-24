import os
import re
import sqlite3
from urllib.parse import quote

import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Local index (build_index.py) replacing live Wikipedia API calls -- deterministic, no rate limits, full control over page ranking.
INDEX_DB_PATH = os.environ.get(
    "FEVER_INDEX_DB",
    os.path.join(os.path.dirname(__file__), "..", "data", "fever_index.db"),
)

_db_conn = None


def _get_db():
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(INDEX_DB_PATH, check_same_thread=False)
    return _db_conn


def _fts_terms(text):
    words = re.findall(r"\w+", text.lower())
    terms = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return terms or words


def lookup_title(title):
    """Exact-title lookup. FEVER/Wikipedia titles use underscores for spaces."""
    conn = _get_db()
    normalized = title.strip().replace(" ", "_")
    row = conn.execute(
        "SELECT title FROM pages WHERE title = ?", (normalized,)
    ).fetchone()
    return row[0] if row else None


# Weight title matches far above intro-text matches, so a page's own subject outranks pages that merely mention it in passing.
_TITLE_WEIGHT, _INTRO_WEIGHT = 5.0, 1.0


def _run_fts(conn, fts_query, limit):
    return conn.execute(
        "SELECT title FROM pages_fts WHERE pages_fts MATCH ? "
        "ORDER BY bm25(pages_fts, ?, ?) LIMIT ?",
        (fts_query, _TITLE_WEIGHT, _INTRO_WEIGHT, limit),
    ).fetchall()


# Disambiguation pages are link lists that densely repeat the query terms, letting them out-rank the real subject page on BM25.
_DISAMBIG_RE = re.compile(r"-LRB-disambiguation-RRB-$")


def search_pages(query, limit=5):
    """Content words required (AND) first for precision; fall back to OR only if that finds nothing."""
    conn = _get_db()
    terms = _fts_terms(query)
    if not terms:
        return []
    quoted = [f'"{t}"' for t in terms]
    rows = _run_fts(conn, " ".join(quoted), limit * 2)
    if not rows:
        rows = _run_fts(conn, " OR ".join(quoted), limit * 2)
    titles = [r[0] for r in rows if not _DISAMBIG_RE.search(r[0])]
    return titles[:limit]


def get_lines(title):
    conn = _get_db()
    row = conn.execute("SELECT lines FROM pages WHERE title = ?", (title,)).fetchone()
    return row[0] if row else ""


NLI_MODEL_NAME = os.environ.get("NLI_MODEL", "roberta-large-mnli")

_tokenizer = None
_model = None
_labels = None


def _load_model():
    global _tokenizer, _model, _labels
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        _model.eval()
        _labels = {k: v.lower() for k, v in _model.config.id2label.items()}
    return _tokenizer, _model, _labels


# FEVER text is Penn-Treebank tokenized (-LRB-/-RRB-, spaced punctuation); clean it up for both NLI input and display.
_DETOK_BRACKETS = {
    "-LRB-": "(", "-RRB-": ")",
    "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}",
}
_DETOK_BRACKET_RE = re.compile("|".join(re.escape(k) for k in _DETOK_BRACKETS))
_DETOK_SPACE_BEFORE_RE = re.compile(r"\s+([.,;:!?'\)\]\}])")
_DETOK_SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[\{])\s+")
_DETOK_DASH_RE = re.compile(r"\s*--\s*")


def detokenize(text):
    text = _DETOK_BRACKET_RE.sub(lambda m: _DETOK_BRACKETS[m.group(0)], text)
    text = _DETOK_SPACE_BEFORE_RE.sub(r"\1", text)
    text = _DETOK_SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = _DETOK_DASH_RE.sub("–", text)
    return re.sub(r"\s+", " ", text).strip()


# Cross-reference lines ("See also...") read like sentences but assert nothing -- filter them out as evidence.
_BOILERPLATE_RE = re.compile(
    r"^(see also|see|main article|further information|for other uses|"
    r"listen|pronunciation|not to be confused)\b",
    re.IGNORECASE,
)
_MYTH_REF_RE = re.compile(r"\bmyth #?\d+\b", re.IGNORECASE)
# Catches disambiguation-list sentences whose title doesn't literally say "(disambiguation)" (e.g. "Great Wall of Mexico").
_MAY_REFER_RE = re.compile(r"\bmay refer to\b", re.IGNORECASE)


def parse_lines(lines_text):
    """FEVER's 'lines' field: one sentence per line, as '<index>\\t<sentence>\\t<hyperlink>...'."""
    sentences = []
    for line in lines_text.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        sent = detokenize(parts[1])
        if len(sent.split()) < 4:
            continue
        if (
            _BOILERPLATE_RE.match(sent)
            or _MYTH_REF_RE.search(sent)
            or _MAY_REFER_RE.search(sent)
        ):
            continue
        sentences.append(sent)
    return sentences


def rank_sentences(claim, candidates, top_n=5, page_rank_decay=0.85):
    """candidates: list of (sentence, title, page_rank); higher-ranked pages break TF-IDF ties, since lexical similarity alone can favor a same-name wrong entity."""
    if not candidates:
        return []
    sentences = [c[0] for c in candidates]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform([claim] + sentences)
    except ValueError:
        return []
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    weighted = [
        (candidates[i][0], candidates[i][1], sims[i] * (page_rank_decay ** candidates[i][2]))
        for i in range(len(candidates))
        if sims[i] > 0
    ]
    weighted.sort(key=lambda item: -item[2])
    return [(sent, title) for sent, title, score in weighted[:top_n]]


# Search every capitalized proper-noun span, not just the leading one -- a full claim's predicate can bias search toward a wrong same-name entity, and multi-subject claims ("Facebook acquired Instagram") need both. Best-effort heuristic, not real NER.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][\w'.]*(?:\s+[A-Z][\w'.]*)*")
_SENTENCE_STARTERS = {"the", "a", "an"}


def extract_entities(claim):
    """Capitalized spans in appearance order; the first is treated as the claim's subject (see retrieve_evidence), so secondary topic words can't substitute for it."""
    claim = claim.strip()
    seen = set()
    entities = []
    for m in _PROPER_NOUN_RE.finditer(claim):
        phrase = m.group(0).strip()
        key = phrase.lower()
        if key in _SENTENCE_STARTERS or key == claim.rstrip(".").lower():
            continue
        if key in seen:
            continue
        seen.add(key)
        entities.append(phrase)
    return entities


def retrieve_evidence(claim, pages_limit=5, sentences_per_claim=5):
    titles = []
    entities = extract_entities(claim)
    if entities:
        subject = entities[0]
        exact = lookup_title(subject)
        subject_hits = search_pages(subject, limit=3)
        if not exact and not subject_hits:
            # Subject doesn't exist in the corpus at all (e.g. a fabricated name) -- don't fall back to secondary topic words, that just finds evidence about the wrong subject.
            return []
        if exact:
            titles.append(exact)
        for title in subject_hits:
            if title not in titles:
                titles.append(title)
        for entity in entities[1:]:
            exact = lookup_title(entity)
            if exact and exact not in titles:
                titles.append(exact)
            for title in search_pages(entity, limit=2):
                if title not in titles:
                    titles.append(title)
    for title in search_pages(claim, limit=pages_limit):
        if title not in titles:
            titles.append(title)

    candidates = []
    for rank, title in enumerate(titles):
        for sent in parse_lines(get_lines(title)):
            candidates.append((sent, title, rank))
    return rank_sentences(claim, candidates, top_n=sentences_per_claim)


def run_nli(pairs):
    """pairs: list of (evidence_sentence, claim). Returns list of {label: prob}."""
    tokenizer, model, labels = _load_model()
    premises = [p[0] for p in pairs]
    hypotheses = [p[1] for p in pairs]
    inputs = tokenizer(
        premises, hypotheses, return_tensors="pt", truncation=True, padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return [
        {labels[i]: float(row[i]) for i in range(row.shape[0])} for row in probs
    ]


# NLI can give a spuriously high-confidence "contradiction" to a barely-relevant sentence, outvoting true entailment on the actual best match; decay-weight by retrieval rank to stop that.
EVIDENCE_RANK_DECAY = 0.8


def verify(claim, entail_threshold=0.55, contradict_threshold=0.55):
    evidence = retrieve_evidence(claim)
    if not evidence:
        return {"verdict": "NOT ENOUGH INFO", "confidence": 0.0, "evidence": []}

    nli_results = run_nli([(sent, claim) for sent, _ in evidence])
    weights = [EVIDENCE_RANK_DECAY**i for i in range(len(nli_results))]

    entail_idx = max(
        range(len(nli_results)),
        key=lambda i: nli_results[i].get("entailment", 0) * weights[i],
    )
    contradict_idx = max(
        range(len(nli_results)),
        key=lambda i: nli_results[i].get("contradiction", 0) * weights[i],
    )
    entail_score = nli_results[entail_idx].get("entailment", 0)
    contradict_score = nli_results[contradict_idx].get("contradiction", 0)
    entail_weighted = entail_score * weights[entail_idx]
    contradict_weighted = contradict_score * weights[contradict_idx]

    if entail_score >= entail_threshold and entail_weighted >= contradict_weighted:
        verdict, confidence, cited_idx = "SUPPORTED", entail_score, entail_idx
    elif contradict_score >= contradict_threshold and contradict_weighted > entail_weighted:
        verdict, confidence, cited_idx = "REFUTED", contradict_score, contradict_idx
    else:
        cited_idx = entail_idx if entail_score >= contradict_score else contradict_idx
        verdict = "NOT ENOUGH INFO"
        confidence = max(entail_score, contradict_score)

    return {
        "verdict": verdict,
        "confidence": round(confidence * 100, 2),
        "cited_sentence": evidence[cited_idx][0],
        "evidence": [
            {
                "sentence": sent,
                "source_title": detokenize(title.replace("_", " ")),
                "url": f"https://en.wikipedia.org/wiki/{quote(detokenize(title.replace('_', ' ')).replace(' ', '_'))}",
            }
            for sent, title in evidence
        ],
    }
