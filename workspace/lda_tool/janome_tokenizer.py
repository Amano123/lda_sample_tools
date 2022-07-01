from typing import List, Tuple, Any
from gensim.corpora import Dictionary
from janome.tokenizer import Tokenizer
from tqdm import tqdm

def janome_tokenizer(sentencese: List[str]) -> List[List[str]]:
    t = Tokenizer(wakati=True)
    words_list: List[List[str]] = []
    s: str
    for s in tqdm(sentencese, total=len(sentencese)):
        words: List[str] = list(t.tokenize(text=s))  # type: ignore
        words_list.append(words)
    return words_list

class make_dictionary(object):
    def __init__(self, sentencese: List[str]):
        self.sentencese = sentencese
        self.dictionary: Dictionary

    def __call__(self):
        pass
