from collections import defaultdict
import dateparser
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
from logmap import logmap
from functools import cached_property
import tomotopy as tp
# from bertopic import BERTopic
# from bertopic.representation import KeyBERTInspired
import pyLDAvis
import orjson
import re
import nltk
from functools import cache, cached_property
from pprint import pprint
tqdm.pandas()



PATH_REPO = os.path.dirname(os.path.dirname(__file__))
PATH_DATA = os.path.join(PATH_REPO,'data')
PATH_STOPWORDS = os.path.join(PATH_DATA,'stopwords.txt')
PATH_METADATA = os.path.join(PATH_DATA,'metadata')
PATH_TXT = os.path.join(PATH_DATA,'txt')
PATH_COMBINED = os.path.join(PATH_DATA,'combined.xlsx')
PATH_NER_DATA=os.path.join(PATH_DATA,'ner_data.sqlitedict')
PATH_GEOLOC_DATA=os.path.join(PATH_DATA,'geocoded_placenames.sqlitedict')

from .utils import *