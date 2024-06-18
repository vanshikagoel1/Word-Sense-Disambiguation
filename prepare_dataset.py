import nltk
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('semcor')
nltk.download('wordnet')
nltk.download('punkt')

# Extract sentences and their corresponding senses
def extract_semcor_data():
    data = []
    for sentence in semcor.sents():
        words = nltk.word_tokenize(" ".join(sentence))
        for word in words:
            senses = semcor.senses(word)
            if senses:
                sense = senses[0].name()  # Take the first sense
                definition = wn.synset(sense).definition()
                data.append({
                    'sentence': " ".join(sentence),
                    'ambiguous_word': word,
                    'sense': definition
                })
    return pd.DataFrame(data)

df = extract_semcor_data()
df.to_csv('semcor_dataset.csv', index=False)
