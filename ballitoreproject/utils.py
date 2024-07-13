from .imports import *


@cache
def get_stopwords():
    with open(PATH_STOPWORDS, "r") as f:
        stopwords = set(f.read().lower().split())
    try:        
        stopwords |= set(nltk.corpus.stopwords.words("english"))
    except Exception:
        nltk.download("stopwords")
        stopwords |= set(nltk.corpus.stopwords.words("english"))
    return stopwords


def tokenize(txt, stopwords=None):
    if stopwords is None:
        stopwords = get_stopwords()

    def remove_bracket_text(text):
        return re.sub(r"\[[^\]]*\]", "", text)

    def is_stopword(word):
        return not word or word.lower() in stopwords or not word[0].isalpha() or word[0].isdigit()

    tokens = re.findall(r"[\w']+|[.,!?; -—–\n]", remove_bracket_text(txt).lower())
    return [word for word in tokens if not is_stopword(word)]


def write_excel(df, path, col_widths=None):
    df.to_excel(path)


def truncfn(fn):
    return "..." + fn[-50:] if len(fn) > 50 else fn


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def default_tokenize(txt):
    tokens = re.findall(r"[\w']+|[.,!?; -—–\n]", txt.lower())
    return tokens
