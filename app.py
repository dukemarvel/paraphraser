import nltk
import spacy
from textblob import Word
from nltk.corpus import wordnet
import string



# Set up spacy
nlp = spacy.load("en_core_web_sm")

def paraphrase(text):
    # Preprocess the text
    text = text.lower()
    punctuation = string.punctuation
    text = ''.join([c for c in text if c not in punctuation])

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Identify the parts of speech
    pos_tags = nltk.pos_tag(tokens)

    # Replace words with synonyms
    paraphrased_tokens = []
    for (word, pos) in pos_tags:
        synonym = find_synonym(word, pos)
        if synonym:
            paraphrased_tokens.append(synonym)
        else:
            paraphrased_tokens.append(word)

    # Assemble the paraphrased text
    paraphrased_text = ' '.join(paraphrased_tokens)
    return paraphrased_text

def find_synonym(word, pos):
    # Use WordNet or a thesaurus to find synonyms for the word
    # based on its part of speech
    synonyms = []
    if pos.startswith('N'):
        synonyms = wordnet.synsets(word, pos='n')
    elif pos.startswith('V'):
        synonyms = wordnet.synsets(word, pos='v')
    elif pos.startswith('J'):
        synonyms = wordnet.synsets(word, pos='a')
    elif pos.startswith('R'):
        synonyms = wordnet.synsets(word, pos='r')

    # If no synonyms are found, return None
    if not synonyms:
        return None

    # Choose the first synonym from the list
    synonym = synonyms[0].lemmas()[0].name()
    return synonym

# Test the paraphraser
text = "The quick brown fox jumps over the lazy dog."
paraphrased_text = paraphrase(text)
print(paraphrased_text)


