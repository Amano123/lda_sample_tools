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
    def __init__(self, dictionary, corpus, output_file_path: str):
        self.dictionary: Dictionary = dictionary
        self.corpus: List[Tuple[int,int]] = corpus
        self.num_topic: int = 5

        self.output_file_path: str = output_file_path
        os.makedirs(self.output_file_path, exist_ok=True)
    
    def lda_traning(self, corpus: List[Tuple[int,int]], num_topic: int, dictionary: Dictionary) -> LdaModel:
        """LDAを学習する"""
        lda_model = LdaModel(
            corpus=corpus, 
            num_topics=num_topic, 
            id2word=dictionary,
            alpha = 'auto',
            eta = 'auto',
            random_state = 0,
            per_word_topics = True
        )
        lda_model.save(f"{self.output_file_path}/lda_modal")
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
        df_topics_contribution.to_csv(f"{self.output_file_path}/topic.csv")

    def dictionary_from_tokent_id_extraction(self, dictionary: Dictionary) -> Tuple[List[str], List[int], Dict[int,int]]:
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
        word_counter: List[int] = []
        for index in range(len(counter_dict)):
            word_counter.append(counter_dict[index])
        return word_counter

    def main(self):
        (self.tokens, # 単語
        self.token_ids, # 単語ID
        self.counter_dict # 単語の出現回数
        ) = self.dictionary_from_tokent_id_extraction(self.dictionary)
        self.word_counter = self.dictionary_from_tokent_cunt_extraction(self.counter_dict)
        # LDAの学習開始
        self.lda_model = self.lda_traning(self.corpus, self.num_topic, self.dictionary)
        # Topicの寄与率を抽出
        self.topics_contribution = self.lda_predicate(self.lda_model)
        self.make_topic_table(self.tokens, self.token_ids, self.topics_contribution, self.word_counter)
        return self.lda_model