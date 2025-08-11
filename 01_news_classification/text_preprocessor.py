# text_preprocessor.py
import nltk, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')  # if you use stopwords

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        # No fitting necessary for preprocessing
        return self

    def transform(self, X):
        return [self._preprocess(text) for text in X]

    def _preprocess(self, text):
        text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
        words = nltk.word_tokenize(text)
        cleaned_words = [
            self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words
        ]
        return ' '.join(cleaned_words)
