from .imports import *
import functools
import json
import smart_open
import orjson
from sqlitedict import SqliteDict

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
        return not word or word.lower() in stopwords or not word[0].isalpha() or word[0].isdigit() or len(word)<3

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

def extract_box_number(s):
    if 'consensus' in s: return 14
    match = re.search(r'b(\d+)', s)
    return int(match.group(1)) if match else 0


def read_json(filename):
    try:
        with open(cache_file, 'rb') as f:
            cache = orjson.loads(f.read())
    except Exception:
        cache = {}
    return cache

CACHED={}
def cached_read_json(filename, force=False):
    global CACHED
    if force or filename not in CACHED:
        CACHED[filename]=read_json(filename)
    return CACHED[filename]



def sqlitedict_cache(filename):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_file = filename
            
            # Create a cache key from the function name and arguments
            key = ' '.join(str(x) for x in args)
            
            # Use sqlitedict for caching
            with SqliteDict(cache_file, autocommit=True) as cache:
                # Check if result is in cache
                if key in cache:
                    return cache[key]
                
                # If not in cache, call the function
                result = func(*args, **kwargs)
                
                # Store result in cache
                cache[key] = result
            
            return result
        return wrapper
    return decorator
