import nltk
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
from collections import Counter
import spacy

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

def importdata():
    url = "https://raw.githubusercontent.com/MaharLeika18/Data-Mining---Python/refs/heads/Loue/Midterms_Act1/MidtermsExam/Jobs.csv"
    text = pd.read_csv(url)

    # Displaying dataset information
    print("Dataset Length: ", len(text))
    print("Dataset Shape: ", text.shape)
    print("First 5 Rows: \n", text.head())
    
    return text

def text_cleaning(text):
    text = text.lower()                 # Force lowercase
    text = re.sub(r'\d+', '', text)     # Remove numbers

    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)   # Remove punctuation
    text = " ".join(text.split())       # Rem whitespace

    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if word.lower()
                     not in stop_words] # Rem stop words

    word_tokens = word_tokenize(text)   # Tokenize
    text = [stemmer.stem(word) for word in word_tokens]

    word_tokens = word_tokenize(text)   # Lemmatization
    text = [lemmatizer.lemmatize(word) for word in word_tokens]

    # Create new column for categorized job positions
    text["categorized_title"] = text["title"]
    
def populate_categ_title(title):
    doc = nlp(title)
    nouns = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    if not nouns:
        return 'other'
    # return most frequent noun
    return Counter(nouns).most_common(1)[0][0]


if __name__ == "__main__":
    df = importdata().text_cleaning()

    df['categorized_title'] = df['title'].apply(populate_categ_title)

    
# Visualization ideas:
# Pie/Bar graph of platforms used to announce job postings
# Pie/Bar of professions