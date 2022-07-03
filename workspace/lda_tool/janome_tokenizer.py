import os 
from typing import List, Tuple, Any, Dict
from gensim.corpora import Dictionary
from gensim import corpora
from gensim.models import TfidfModel
from janome.tokenizer import Tokenizer
from tqdm import tqdm

def janome_tokenizer(sentencese: List[str]) -> List[List[str]]:
    t = Tokenizer()
    words_list: List[List[str]] = []
    s: str
    for s in tqdm(sentencese, total=len(sentencese)):
        words: List[str] = []
        for token in t.tokenize(text=s):# type: ignore
            if (token.part_of_speech.split(',')[0] == '名詞' # type: ignore
            and token.part_of_speech.split(',')[1] in ['一般']):# type: ignore
                words.append(token.surface)  # type: ignore
        words_list.append(words)
    return words_list

def ginza_tokenizer(sentencese: List[str]) -> List[List[str]]:
    return 0
    pass


class Preprocess(object):
    def __init__(self, words_list: List[List[str]], output_file_path: str):
        self.output_file_path: str = output_file_path
        os.makedirs(self.output_file_path, exist_ok=True)

        self.words_list = words_list
        self.dictionary: Dictionary = self.make_dictionary(words_list)
        self.corpus = self.make_corpus(words_list, self.dictionary)
        self.tf_idf_corpus = self.make_tf_idf_corpus(self.corpus)
    
    def make_dictionary(self, words_list: List[List[str]]) -> Dictionary:
        """辞書を作成"""
        dictionary = corpora.Dictionary(words_list)
        dictionary.save(f'{self.output_file_path}/dict.dict')
        dictionary.save_as_text(f'{self.output_file_path}/dict.dict.txt')
        return dictionary
    
    def make_tf_idf_corpus(self, corpus):
        test_model = TfidfModel(corpus)
        corpus_tfidf = test_model[corpus]
        return corpus_tfidf
    
    def make_corpus(self, words_list: List[List[str]], dictionary: Dictionary) -> List[Tuple[int,int]]:
        """corpusを作成"""
        corpus: List[Tuple[int,int]] = [
            dictionary.doc2bow(text) for text in words_list
            ]
        return corpus