import streamlit as st
import random

# Define the dataset
dataset = [
    {
        "sentence": "He saw a bat flying in the sky.",
        "ambiguous_word": "bat",
        "meanings": ["a flying mammal", "a piece of sports equipment", "to open and close your eyes quickly"],
        "correct_meaning": "a flying mammal"
    },
    {
        "sentence": "She hit the ball with a bat.",
        "ambiguous_word": "bat",
        "meanings": ["a flying mammal", "a piece of sports equipment", "to open and close your eyes quickly"],
        "correct_meaning": "a piece of sports equipment"
    },
    {
        "sentence": "The bank charges a fee for ATM withdrawals.",
        "ambiguous_word": "bank",
        "meanings": ["a financial institution", "the land alongside or sloping down to a river or lake", "a row of similar things", "to tilt or bend to one side"],
        "correct_meaning": "a financial institution"
    },
    {
        "sentence": "The fisherman sat by the bank of the river, waiting for a bite.",
        "ambiguous_word": "bank",
        "meanings": ["a financial institution", "the land alongside or sloping down to a river or lake", "a row of similar things", "to tilt or bend to one side"],
        "correct_meaning": "the land alongside or sloping down to a river or lake"
    },
    {
        "sentence": "She arranged the books neatly in a bank on the shelf.",
        "ambiguous_word": "bank",
        "meanings": ["a financial institution", "the land alongside or sloping down to a river or lake", "a row of similar things", "to tilt or bend to one side"],
        "correct_meaning": "a row of similar things"
    },
    {
        "sentence": "The boat started to bank as it turned sharply.",
        "ambiguous_word": "bank",
        "meanings": ["a financial institution", "the land alongside or sloping down to a river or lake", "a row of similar things", "to tilt or bend to one side"],
        "correct_meaning": "to tilt or bend to one side"
    },
    {
        "sentence": "She put the flowers in a vase on the table.",
        "ambiguous_word": "vase",
        "meanings": ["a container for holding flowers or other plants", "an anatomical vessel that carries blood throughout the body", "a decorative piece for displaying flowers", "a place for depositing or storing something"],
        "correct_meaning": "a container for holding flowers or other plants"
    },
    {
        "sentence": "The bride held a bouquet of roses in her hands.",
        "ambiguous_word": "bouquet",
        "meanings": ["a decorative bunch of flowers, especially one presented as a gift or carried at a ceremony", "a traditional French bread", "a cluster of balloons or decorative items", "a group of people or things"],
        "correct_meaning": "a decorative bunch of flowers, especially one presented as a gift or carried at a ceremony"
    },
    {
        "sentence": "The bakery sold a fresh bouquet of baguettes every morning.",
        "ambiguous_word": "bouquet",
        "meanings": ["a decorative bunch of flowers, especially one presented as a gift or carried at a ceremony", "a traditional French bread", "a cluster of balloons or decorative items", "a group of people or things"],
        "correct_meaning": "a traditional French bread"
    }
]

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

    st.write("Choose the correct sense for the word '{}' :".format(question['ambiguous_word']))
    st.session_state.selected_option = st.radio("", question['meanings'], index=None)

    if st.session_state.selected_option is not None:
        if st.button("Check Answer"):
            if st.session_state.selected_option == question['correct_meaning']:
                st.success("Congratulations! You chose the correct sense.")
            else:
                st.error("Sorry, that's incorrect. The correct sense is '{}'.".format(question['correct_meaning']))
            st.session_state.selected_option = None  # Reset selected option
            st.button("Next Question")
            st.session_state.question_index = random.randint(0, len(dataset) - 1)

if __name__ == "__main__":
    main()