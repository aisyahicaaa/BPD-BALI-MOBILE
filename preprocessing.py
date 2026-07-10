import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (hanya sekali saat pertama dijalankan)
nltk.download("stopwords")

# ==========================
# DATA CLEANING
# ==========================
def text_cleaning(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Hapus URL
    text = re.sub(r"http\S+|www\S+", "", text)

    # Hapus mention & hashtag
    text = re.sub(r"@\w+|#\w+", "", text)

    # Hapus angka
    text = re.sub(r"\d+", "", text)

    # Hapus tanda baca
    text = re.sub(r"[^\w\s]", " ", text)

    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================
# CASE FOLDING
# ==========================
def casefolding(text):
    return text.lower()


# ==========================
# TOKENIZING
# ==========================
def tokenize_text(text):

    if not isinstance(text, str):
        return []

    return text.split()


# ==========================
# NORMALISASI
# ==========================
lexicon_url = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"

lexicon_df = pd.read_csv(lexicon_url)

norm_dict = dict(zip(lexicon_df["slang"], lexicon_df["formal"]))


def normalisasi_term(word_list):

    if not isinstance(word_list, list):
        return []

    return [norm_dict.get(word, word) for word in word_list]


# ==========================
# STOPWORD REMOVAL
# ==========================
STOPWORDS = set(stopwords.words("indonesian"))

custom_stopwords = {
    'yang','aja','nya','kok',
    'untuk','dan','di','ke',
    'ini','ya','nih','dong',
    'sih','deh','lah','e'
}

STOPWORDS.update(custom_stopwords)


def remove_stopwords(word_list):

    if not isinstance(word_list, list):
        return []

    return [word for word in word_list if word not in STOPWORDS]


# ==========================
# PREPROCESSING UTAMA
# ==========================
def preprocess_text(text):

    text = text_cleaning(text)
    text = casefolding(text)
    text = tokenize_text(text)
    text = normalisasi_term(text)
    text = remove_stopwords(text)

    return " ".join(text)
