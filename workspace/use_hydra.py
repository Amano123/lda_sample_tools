import os
import hydra
from omegaconf import DictConfig, OmegaConf
import LDA_main

@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg : DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))

    dataset_path = cfg.dataset.text_file_path
    num_topic = cfg.LDA.topic_num
    lda_model = LDA_main.main(dataset_path, num_topic)


if __name__ == "__main__":
    hydra_main()