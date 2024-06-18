#dividing model,game logiv ND app
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Download necessary resources (assuming not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Preprocess function to remove noise from text
def preprocess_text(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# Function to get possible meanings of an ambiguous word in a sentence
def get_meanings(sentence, word):
    synsets = wn.synsets(word)
    meanings = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            meanings.add(lemma.name())
    return list(meanings)

# Function to extract additional features
def extract_additional_features(sentence, ambiguous_word):
    features = {}
    tokens = word_tokenize(sentence)
    word_index = tokens.index(ambiguous_word)
    # Get part-of-speech (POS) tag for the ambiguous word
    features["pos_tag"] = nltk.pos_tag(tokens)[word_index][1]
    # Get bigrams (two consecutive words) around the ambiguous word
    if word_index > 0:
        features["bigram_before"] = tokens[word_index - 1]
    if word_index < len(tokens) - 1:
        features["bigram_after"] = tokens[word_index + 1]
    return features

# Define the dataset with sentences
dataset = [
    {"sentence": "He saw a bat flying in the sky.", "ambiguous_word": "bat", "label": "animal"},
    {"sentence": "She hit the ball with a bat.", "ambiguous_word": "bat", "label": "tool"},
    {"sentence": "The vampire bat feeds on blood.", "ambiguous_word": "bat", "label": "animal"},
    {"sentence": "She used a baseball bat to hit the ball.", "ambiguous_word": "bat", "label": "tool"},
    {"sentence": "The bank approved the loan application.", "ambiguous_word": "bank", "label": "financial institution"},
    {"sentence": "We sat by the bank of the lake and enjoyed the view.", "ambiguous_word": "bank", "label": "shoreline"},
    {"sentence": "The river bank collapsed after heavy rainfall.", "ambiguous_word": "bank", "label": "riverbank"},
    {"sentence": "He deposited his money in the bank.", "ambiguous_word": "bank", "label": "financial institution"},
    {"sentence": "She tied the ribbon into a bow.", "ambiguous_word": "bow", "label": "knot"},
    {"sentence": "He aimed his bow at the target.", "ambiguous_word": "bow", "label": "weapon"},
    {"sentence": "The fisherman sat by the bank of the river, waiting for a bite.", "ambiguous_word": "bank", "label": "riverbank"}
    # ... (other examples from original dataset)
]

# Prepare the dataset with features
X = []
y = []
for example in dataset:
    meanings = get_meanings(example['sentence'], example['ambiguous_word'])
    for meaning in meanings:
        sentence_features = extract_additional_features(example['sentence'], example['ambiguous_word'])
        # Combine sentence (already preprocessed) with additional features
        features = {"sentence": preprocess_text(example['sentence']), **sentence_features}
        X.append(features)
        y.append(example['label'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform([x["sentence"] for x in X_train])
X_test_vec = vectorizer.transform([x["sentence"] for x in X_test])

# Train the model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate the model accuracy
accuracy = model.score(X_test_vec, y_test)
print("Accuracy:", accuracy)

# Save the trained model
joblib.dump(model, 'word_sense_disambiguation_model.pkl')
print("Model saved successfully.")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Vectorizer saved successfully.")
