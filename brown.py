import nltk
from nltk.corpus import brown

# Download necessary resources (assuming not already downloaded)
nltk.download('brown')
nltk.download('punkt')

# Function to extract instances from the Brown corpus
def extract_brown_instances():
    brown_instances = []
    ambiguous_words = ["bank", "bat"]  # Example ambiguous words
    for category in brown.categories():
        for fileid in brown.fileids(category):
            sentences = brown.sents(fileid)
            for sentence in sentences:
                # Check if the sentence contains an ambiguous word
                for word in sentence:
                    if word.lower() in ambiguous_words:
                        # Extract the sentence containing the ambiguous word
                        sentence_text = ' '.join(sentence)
                        # Determine the label based on the context (e.g., POS tagging)
                        # You may need additional processing here to determine the label
                        label = "unknown"  # Placeholder label
                        # Append the instance to the list of brown_instances
                        brown_instances.append({"sentence": sentence_text, "ambiguous_word": word.lower(), "label": label})
    return brown_instances

# Extract instances from the Brown corpus
brown_instances = extract_brown_instances()

# Display a sample of extracted instances
for instance in brown_instances[:5]:
    print(instance)
