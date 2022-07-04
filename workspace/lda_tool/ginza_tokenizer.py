#%%
from typing import List, Tuple, Any, Dict
import spacy
from spacy import Language
from tqdm import tqdm

class Ginza_tokenizer():
    def __init__(self, spacy_model: str = 'ja_ginza'):
        self.nlp: Language = spacy.load(spacy_model)

    def tokenize(self, sentencese: List[str]):#-> List[List[str]]:
        """
        文章を形態素解析する関数
        Input:
            - sentencese : ["銀座でランチをご一緒しましょう。", "今日は最高の1日になりそうだ"]
        Return:
            - words_list : [['銀座', 'ランチ', '一緒'], ['今日', '最高', '1日']]
        """
        words_list: List[List[str]] = []
        for sentence in tqdm(sentencese, total=len(sentencese)):
            word_list: List[str] = []
            doc = self.nlp(sentence)
            for sent in doc.sents:
                for token in sent:
                    # if  "名詞" in token.tag_:
                    if token.pos_ in ["NOUN", "PROPN"]:
                        word_list.append(token.text)
            words_list.append(word_list)
        return words_list


# %%
if __name__ == "__main__":
    sentencese = ["銀座でランチをご一緒しましょう。", "今日は最高の1日になりそうだ"]
    g_tokenizer = Ginza_tokenizer()
    words_list = g_tokenizer.tokenize(sentencese)
    print(words_list)
# %%
