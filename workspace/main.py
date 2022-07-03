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

now = datetime.now()
output_file_path = f'./output/{now.strftime("%Y-%m-%d/%H-%M-%S")}'

# %%
with open("./sample_dataset/cooking.txt", mode="r") as f:
    dataset = f.read().split("\n")

#%%
g_tokenizer = Ginza_tokenizer()
word_list = g_tokenizer.tokenize(dataset)
# %%
# word_list = janome_tokenizer(dataset)
preprocess = Preprocess(word_list, output_file_path)

dictionary = preprocess.dictionary
corpus = preprocess.corpus

#%%
corpus_tfidf = preprocess.tf_idf_corpus

#%%
lda_tool = LDA_Tool(dictionary, corpus, output_file_path)
lda_model = lda_tool.main()

# %%
