#%%
import os
from typing import Any, List, Tuple
from tqdm import tqdm

from janome.tokenizer import Tokenizer

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import TfidfModel

from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra


#%%
class janome_tokenizer():
    def __init__(self) -> None:
        self.word_tokenizer: Tokenizer = Tokenizer()
    
    def pos_tokenizer(self, sentence: str) -> List[str]:
        tokens: List[str] = []
        # TODO:品詞情報を取れるようにする
        for token in self.word_tokenizer.tokenize(sentence):
            tokens.append(token.surface)
        return tokens   
 
    def pos_tokenizer_list(self, sentence: List[str]) -> List[List[str]]:
        """
        in : List[str]
        out : List[List[str]]
        """
        token_list: List[List[str]] = []
        for text in sentence:
            token_list.append(
                [
                    token
                    for token in self.word_tokenizer.tokenize(text)
                ]
            )
        return token_list
    
    def noun_tokenizer_list(self, sentence: List[str]) -> List[List[str]]:
        token_list: List[List[str]] = []
        for text in sentence:
            token_list.append(
                [
                    token.surface 
                    for token in self.word_tokenizer.tokenize(text)
                    if token.part_of_speech.split(',')[0] in ['名詞'] 
                        and token.part_of_speech.split(',')[1] in ['一般']   
                ]
            )
        # 空配列削除
        token_list = [i for i in token_list if i]
        # 1単語の配列削除
        token_list = [i for i in token_list if len(i) > 1]

        return token_list

#%%
class LDA():
    def __init__(
        self, 
        cfg,
        lda_use_tokenizer: janome_tokenizer, 
        ) -> None:
        self.original_path = hydra.utils.get_original_cwd()
        self.tokenizer: janome_tokenizer= lda_use_tokenizer
        self.text_file_path: str= "dataset/" + cfg.dataset.text_file_path
        self.save_file_path: str = cfg.save_file.save_file_path
        os.makedirs(self.save_file_path, exist_ok=True)
        self.topic_file_path: str = f"{self.save_file_path}/topic"
        os.makedirs(self.topic_file_path, exist_ok=True)
        self.text_data: List[str] = open(self.original_path + "/" + self.text_file_path).readlines()
        self.token_list: List[List[str]] = self.tokenizer.noun_tokenizer_list(self.text_data)
        with open(f"{self.save_file_path}/token_list.csv", mode="w") as f:
            f.write("\n".join(
                [
                    ",".join(i) 
                    for i in self.token_list
                ]
            )
        )
        self.topic_start: int = cfg.lda_parameter.topic_start
        self.topic_limit: int = cfg.lda_parameter.topic_limit
        self.topic_step: int = cfg.lda_parameter.topic_step
    
    def make_dictionary(self) -> Dictionary:
        #word - id 辞書作成
        dictionary: Dictionary = Dictionary(self.token_list)
        dictionary.filter_extremes(no_below=1, no_above=0.5)
        print("Number of unique tokens: %d" % len(dictionary))
        return dictionary

    def make_corpus(self, dictionary: Dictionary) -> List[Any]:
        # corpus作成
        corpus: List[Any] = [
            dictionary.doc2bow(token) 
            for token in self.token_list
        ]
        print("Number of documents: %d" % len(corpus))
        return corpus
    
    def make_tf_idf(self, corpus: List[Any]):
        # tfidf作成
        tfidf_model: TfidfModel = TfidfModel(corpus)
        corpus_tfidf = tfidf_model[corpus]
        return corpus_tfidf

    def make_lda_model(self, n_topic: int) -> LdaMulticore:
        lda_model: LdaMulticore = LdaMulticore(
            corpus=self.corpus_tfidf, 
            id2word=self.dictionary, 
            num_topics=n_topic, 
            random_state=0,
            alpha="symmetric",
            eta="auto"
        )
        return lda_model
    
    def lda_score_estimation(self, model: LdaMulticore, n_topic: int) -> None :
        top_topics: List[Tuple[Any, Any]] = model.top_topics(self.corpus_tfidf)
        avg_topic_coherence: float = sum([t[1] for t in top_topics]) / n_topic
        print("Average topic coherence: %.4f." % avg_topic_coherence)

    def save_topic_words(self, model: LdaMulticore, n_topic: int) -> None:
        topic_word = []
        for i, t in enumerate(range(model.num_topics)):
            x = dict(model.show_topic(t, 30))
            topic_word.append([f"{i}" for i in x])
        T_topic_word: List[Any] = list(zip(*topic_word))
        with open(f"{self.topic_file_path}/{n_topic}_topic_word.txt", mode="w") as f:
            for i in T_topic_word:
                f.write("\t".join(i))
                f.write("\n")

    def main(self) -> None:
        self.dictionary: Dictionary = self.make_dictionary()
        self.corpus: List[Any] = self.make_corpus(self.dictionary)
        self.corpus_tfidf = self.make_tf_idf(self.corpus)
        topic_range: range = range(self.topic_start, self.topic_limit, self.topic_step)

        for n_topic in tqdm(topic_range):
            self.lda_model: LdaMulticore = self.make_lda_model(n_topic)
            self.lda_score_estimation(self.lda_model, n_topic)
            self.save_topic_words(self.lda_model, n_topic)
            
#%%
@hydra.main(config_name="config", config_path="config")
def main(cfg: DictConfig):
    tokenizer = janome_tokenizer()
    lda = LDA(cfg,tokenizer)
    lda.main()

#%%
if __name__ == '__main__':
    main()
# %%
