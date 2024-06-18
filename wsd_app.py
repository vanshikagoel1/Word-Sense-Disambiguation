import random
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from pywsd.lesk import simple_lesk
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import wordnet as wn

# Download necessary resources
nltk.download('punkt')
nltk.download('wordnet')

# Define the dataset with sentences
dataset = [
    {"sentence": "He saw a bat flying in the sky.", "ambiguous_word": "bat"},
    {"sentence": "She hit the ball with a bat.", "ambiguous_word": "bat"},
    {"sentence": "The bank charges a fee for ATM withdrawals.", "ambiguous_word": "bank"},
    {"sentence": "The fisherman sat by the bank of the river, waiting for a bite.", "ambiguous_word": "bank"},
    {"sentence": "She arranged the books neatly in a bank on the shelf.", "ambiguous_word": "bank"},
    {"sentence": "The boat started to bank as it turned sharply.", "ambiguous_word": "bank"},
    {"sentence": "She put the flowers in a vase on the table.", "ambiguous_word": "vase"},
    {"sentence": "The bride held a bouquet of roses in her hands.", "ambiguous_word": "bouquet"},
    {"sentence": "The bark of the tree was rough.", "ambiguous_word": "bark"},
    {"sentence": "She saw the light at the end of the tunnel.", "ambiguous_word": "light"},
    {"sentence": "He wore a tie to the meeting.", "ambiguous_word": "tie"}
]

def get_meanings(sentence, word):
    # Convert list of tokens to a single string
    sentence_str = ' '.join(sentence)
    
    # Use Lesk algorithm to disambiguate the sense of the word based on the sentence
    sense = simple_lesk(sentence_str, word)
    
    extras = {}
    for ss in wn.synsets(word):
        if ss.definition() != sense.definition():
            extras[ss.definition()] = False
        
        if len(extras.keys()) >= 3:
            break

    if sense:
        extras[sense.definition()] = True

        return extras
    else:
        return "No suitable meaning found."


def main():
    st.set_page_config(page_title="Word Sense Disambiguation Game", layout="wide")

    if 'question_index' not in st.session_state:
        st.session_state.question_index = random.randint(0, len(dataset) - 1)
        st.session_state.selected_option = None

    question = dataset[st.session_state.question_index]

    st.sidebar.title("Instructions")
    st.sidebar.write("Choose the correct sense based on the context sentence.")

    st.info("Context Sentence:")
    st.write(question['sentence'])

    # Get possible meanings dynamically using Lesk algorithm
    meaning_map = get_meanings(question['sentence'], question['ambiguous_word'])

    for meaning in meaning_map:
        if meaning_map[meaning]:
            correct_meaning = meaning
    
    # Show options
    st.write("Choose the correct sense for the word '{}' :".format(question['ambiguous_word']))
    st.session_state.selected_option = st.radio("", meaning_map.keys(), index=None)

    if st.session_state.selected_option is not None:
        if st.button("Check Answer"):
            if st.session_state.selected_option == correct_meaning:
                st.success("Congratulations! You chose the correct sense.")
            else:
                st.error("Sorry, that's incorrect. The correct sense is '{}'.".format(correct_meaning))
            st.session_state.selected_option = None  # Reset selected option
            st.button("Next Question")
            st.session_state.selected_option = None  # Reset selected option
            st.session_state.question_index = random.randint(0, len(dataset) - 1)

if __name__ == "__main__":
    main()
