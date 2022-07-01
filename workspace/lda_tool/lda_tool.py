from typing import List, Tuple, Dict, Union
import os
import gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import pandas as pd
import numpy as np
from datetime import datetime

class LDA_Tool(object):
    def __init__(self, word_list: List[List[str]]):
        self.word_list: List[List[str]] =  word_list
        self.dictionary: Dictionary
        self.corpus: List[Tuple[int,int]]
        self.lda_model: LdaModel
        self.num_topic: int = 5
        self.tokens: List[str]
        self.token_ids: List[int]
        self.counter_dict: Dict[int, int]
        self.topics_contribution: List[List[float]]
        now = datetime.now()
        self.output_file_path: str = \
            f'./output/{now.strftime("%Y-%m-%d/%H-%M-%S")}'
        os.makedirs(self.output_file_path, exist_ok=True)

    def make_dictionary(self, words_list: List[List[str]]) -> Dictionary:
        # ! corpus作成はこのクラスに必要ない？
        """辞書を作成"""
        dictionary = corpora.Dictionary(words_list)
        dictionary.save(f'{self.output_file_path}/cooking.dict')
        dictionary.save_as_text(f'{self.output_file_path}/cooking.dict.txt')
        return dictionary
    
    def make_corpus(self, words_list: List[List[str]], dictionary: Dictionary) -> List[Tuple[int,int]]:
        # ! corpus作成はこのクラスに必要ない？
        """corpusを作成"""
        corpus: List[Tuple[int,int]] = [
            dictionary.doc2bow(text) for text in words_list
            ]
        return corpus
    
    def dictionary_from_tokent_id_extraction(self, dictionary: Dictionary) -> Tuple[List[str], List[int], Dict[int,int]]:
        #! 必要ない？
        """辞書から単語と単語IDと出現回数を抽出"""
        # 辞書（Dictionary）から、単語とIDのペアを抽出
        tuple_token2id: List[Tuple[str,int]] = list(dictionary.token2id.items())
        token: List[str]
        token_id : List[int]
        # 転置を使って単語リスト、IDリストを作成
        token, token_id = list(zip(*tuple_token2id))  # type: ignore
        # 単語の出現回数を抽出
        counter_dict: Dict[int,int] = dictionary.cfs
        return (token, token_id, counter_dict)
    
    def dictionary_from_tokent_cunt_extraction(self, counter_dict: Dict[int,int]) -> List[int]:
        #! 必要ない？
        word_counter: List[int] = []
        for index in range(len(counter_dict)):
            word_counter.append(counter_dict[index])
        return word_counter 
    
    def lda_traning(self, corpus: List[Tuple[int,int]], num_topic: int, dictionary: Dictionary) -> LdaModel:
        """LDAを学習する"""
        lda_model = LdaModel(
            corpus=corpus, 
            num_topics=num_topic, 
            id2word=dictionary
            )
        return lda_model

    def lda_predicate(self, lda_model: LdaModel) -> List[List[float]]:
        # topics情報を抽出：このままだと numpy array
        topics_numpy_array = lda_model.get_topics() 
        # numpy arrayをlistに変換する
        # [Topic][word]
        topics_contribution: List[List[float]] = topics_numpy_array.tolist()
        return topics_contribution
    
    def make_topic_table(self, token: List[str], token_id: List[int], topics_contribution: List[List[float]], word_counter: List[int]):
        df_data = {}
        df_data["token_id"] = token_id
        df_data["token"] = token
        df_data["cunt"] = word_counter
        for topic_num, topic in enumerate(topics_contribution, 1):
            df_data[f"topic_{topic_num}"] = topic
        df_topics_contribution = pd.DataFrame(df_data)
        df_topics_contribution.to_csv(f"{self.output_file_path}/test.csv")

    def main(self):
        # 辞書作成
        self.dictionary = self.make_dictionary(self.word_list)
        # 単語、単語ID、出現回数を抽出
        (
        self.tokens, # 単語
        self.token_ids, # 単語ID
        self.counter_dict # 単語の出現回数
        ) = self.dictionary_from_tokent_id_extraction(self.dictionary)
        # 単語の回数
        self.word_counter = self.dictionary_from_tokent_cunt_extraction(self.counter_dict)
        # corpus作成
        self.corpus = self.make_corpus(self.word_list, self.dictionary)
        # LDAの学習開始
        self.lda_model = self.lda_traning(self.corpus, self.num_topic, self.dictionary)
        # Topicの寄与率を抽出
        self.topics_contribution = self.lda_predicate(self.lda_model)
        self.make_topic_table(self.tokens, self.token_ids, self.topics_contribution, self.word_counter)