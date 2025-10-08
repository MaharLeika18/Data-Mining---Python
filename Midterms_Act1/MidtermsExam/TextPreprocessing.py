import nltk
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def importdata():
    url = ""
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

    