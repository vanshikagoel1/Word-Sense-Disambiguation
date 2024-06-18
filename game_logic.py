import random
import joblib
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet as wn

# Download necessary resources (assuming not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Load the trained model and TF-IDF vectorizer
model = joblib.load('word_sense_disambiguation_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocess function to remove noise from text
def preprocess_text(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# Function to implement the simplified Lesk algorithm
def lesk_algorithm(context_sentence, ambiguous_word):
    context = set(word_tokenize(context_sentence))
    synsets = wn.synsets(ambiguous_word)
    if not synsets:
        return None
    best_sense = synsets[0]
    max_overlap = 0
    for sense in synsets:
        gloss = set(word_tokenize(sense.definition()))
        examples = set(word_tokenize(' '.join(sense.examples())))
        signature = gloss.union(examples)
        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    return best_sense

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

# Define the dataset with questions
dataset = [
    {"sentence": "He saw a bat flying in the sky.", "ambiguous_word": "bat", "options": ["animal", "tool", "sports", "building"], "correct": "animal"},
    {"sentence": "She hit the ball with a bat.", "ambiguous_word": "bat", "options": ["animal", "tool", "sports", "building"], "correct": "tool"},
    {"sentence": "The fisherman sat by the bank of the river, waiting for a bite.", "ambiguous_word": "bank", "options": ["a financial institution", "the land alongside or sloping down to a river or lake", "a row of similar things", "to tilt or bend to one side"], "correct": "the land alongside or sloping down to a river or lake"},
    {"sentence": "The bride held a bouquet of roses in her hands.", "ambiguous_word": "bouquet", "options": ["a decorative bunch of flowers, especially one presented as a gift or carried at a ceremony", "a traditional French bread", "a cluster of balloons or decorative items", "a group of people or things"], "correct": "a decorative bunch of flowers, especially one presented as a gift or carried at a ceremony"},
    {"sentence": "The actor delivered his lines with perfect timing.", "ambiguous_word": "actor", "options": ["profession", "device", "measurement", "fictional character"], "correct": "profession"},
    {"sentence": "The tree fell across the road, blocking traffic.", "ambiguous_word": "tree", "options": ["plant", "structure", "animal", "element"], "correct": "plant"}
    # ... (more questions)
]

# Function to get a new question
def get_new_question():
    return random.choice(dataset)

# Function to check the answer
def check_answer(selected_option, correct_option):
    return selected_option == correct_option
