#%%
from typing import List, Tuple, Dict
import gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from tqdm import tqdm
import pandas as pd

from lda_tool import janome_tokenizer
from lda_tool import LDA_Tool
# %%
with open("./sample_dataset/cooking.txt", mode="r") as f:
    dataset = f.read().split("\n")

# %%
word_list = janome_tokenizer(dataset)
#%%
lda_tool = LDA_Tool(word_list)
df = lda_tool.main()
# %%
