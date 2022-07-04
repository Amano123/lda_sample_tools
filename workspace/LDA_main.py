#%%
from typing import List, Tuple, Dict
import gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from lda_tool import janome_tokenizer
from lda_tool import Ginza_tokenizer
from lda_tool import Preprocess
from lda_tool import LDA_Tool

def main(dataset_path: str, num_topic: int):
    now = datetime.now()
    # output_file_path = f'./output/{now.strftime("%Y-%m-%d/%H-%M-%S")}'

    with open(dataset_path, mode="r") as f:
        dataset = f.read().split("\n")

    g_tokenizer = Ginza_tokenizer()
    word_list = g_tokenizer.tokenize(dataset)

    # word_list = janome_tokenizer(dataset)
    preprocess = Preprocess(word_list)

    dictionary = preprocess.dictionary
    corpus = preprocess.corpus

    #%%
    corpus_tfidf = preprocess.tf_idf_corpus

    #%%
    lda_tool = LDA_Tool(dictionary, corpus, num_topic)
    lda_model = lda_tool.main()

# %%
if __name__ == "__main__":
    dataset_path: str = "/home/amano/workspace/sample_dataset/2019-02-15_train.txt"
    num_topic: int = 10
    main(dataset_path, num_topic)