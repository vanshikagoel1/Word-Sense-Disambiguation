import streamlit as st
import joblib
import random
from nltk.tokenize import word_tokenize
import nltk

# Download necessary resources (assuming not already downloaded)
nltk.download('punkt')

# Load the trained model and TF-IDF vectorizer
model = joblib.load('word_sense_disambiguation_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocess function to remove noise from text
def preprocess_text(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens)

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

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Word Sense Disambiguation Game", layout="wide")

    if 'question_index' not in st.session_state:
        st.session_state.question_index = random.randint(0, len(dataset) - 1)
        st.session_state.selected_option = None
        st.session_state.score = 0
        st.session_state.feedback = ""
        st.session_state.display_next_button = False

    question = dataset[st.session_state.question_index]

    st.sidebar.title("Instructions")
    st.sidebar.write("Choose the correct sense based on the context sentence.")

    st.title("Word Sense Disambiguation Game")

    st.info("Context Sentence:")
    st.write(f"**{question['sentence']}**")

    st.write(f"Choose the correct sense for the word '**{question['ambiguous_word']}**':")
    selected_option = st.radio("", question['options'], index=None)

    if selected_option is not None:
        st.session_state.selected_option = selected_option

    if st.button("Check Answer"):
    # Ensure an option is selected
       if st.session_state.selected_option is not None:
        # Vectorize the context sentence
        context_vector = vectorizer.transform([preprocess_text(question['sentence'])])
        # Predict using the model
        predicted_label = model.predict(context_vector)[0]
        predicted_label = predicted_label.lower()  # Convert to lowercase
        # Convert all options to lowercase
        options_lower = [option.lower() for option in question['options']]
        if predicted_label in options_lower:
            predicted_index = options_lower.index(predicted_label)
            predicted_sense = question['options'][predicted_index]

            if predicted_sense == question['correct']:
                st.success("Congratulations! You chose the correct sense.")
                st.session_state.score += 1
                st.session_state.feedback = "Correct!"
            else:
                st.error(f"Sorry, that's incorrect. The correct sense is '{question['correct']}'.")
                st.session_state.feedback = f"Incorrect! The correct sense is '{question['correct']}'."

        else:
            st.error(f"Error: The predicted label '{predicted_label}' is not in the available options.")

        # Load a new question and reset the selected option
        st.session_state.question_index = random.randint(0, len(dataset) - 1)
        st.session_state.selected_option = None
        st.session_state.display_next_button = True
    else:
        st.error("Please select an option before checking the answer.")



    if st.session_state.display_next_button:
        if st.button("Next Question"):
            st.session_state.display_next_button = False
            st.session_state.feedback = ""

    if st.session_state.feedback:
        st.write(st.session_state.feedback)

    st.write(f"**Score: {st.session_state.score}**")

if __name__ == "__main__":
    main()
