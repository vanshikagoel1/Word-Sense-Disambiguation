import streamlit as st
import random
import game_logic
import nltk

# Download necessary resources (assuming not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Word Sense Disambiguation Game", layout="wide")

    if 'question_index' not in st.session_state:
        st.session_state.question_index = random.randint(0, len(game_logic.dataset) - 1)
        st.session_state.selected_option = None
        st.session_state.score = 0
        st.session_state.feedback = ""
        st.session_state.display_next_button = False

    question = game_logic.dataset[st.session_state.question_index]

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
        if game_logic.check_answer(st.session_state.selected_option, question['correct']):
            st.success("Congratulations! You chose the correct sense.")
            st.session_state.score += 1
            st.session_state.feedback = "Correct!"
        else:
            st.error(f"Sorry, that's incorrect. The correct sense is '{question['correct']}'.")
            st.session_state.feedback = f"Incorrect! The correct sense is '{question['correct']}'."

        # Load a new question and reset the selected option
        st.session_state.question_index = random.randint(0, len(game_logic.dataset) - 1)
        st.session_state.selected_option = None
        st.session_state.display_next_button = True

    if st.session_state.display_next_button:
        if st.button("Next Question"):
            st.session_state.display_next_button = False
            st.session_state.feedback = ""

    if st.session_state.feedback:
        st.write(st.session_state.feedback)

    st.write(f"**Score: {st.session_state.score}**")

if __name__ == "__main__":
    main()
