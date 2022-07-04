# <div style="text-align: center">LDA Sample Tools</div>

Latent Dirichlet Allocationと呼ばれるトピックモデルのサンプルプログラムを作成  

## 構成

|          |               |  
| -------- | :------------ |  
| python   | 3.8         |  
| lda      | gensim 4.0.1  |  
| analyzer | Janome 0.3.10 |  

## 分析データ

datasetの中にサンプルファイルとして

- cooking.txt
  - Wikipedia内の料理関係の記事（500文）
- poli.txt
  - Wikipedia内の高分子関係の記事（500文）

を置いています。

datasetに追加で用意すれば分析できるようにしています。

## 実行

2021/08/19現在

```shell
docker-compose build
docker-compose up -d
```

## hydraの設定ファイル  

config/config.yaml

```yaml
dataset:
  text_file_path: "poli.txt" #ここで分析ファイルを指定する。
save_file:
  save_file_path: "output"
lda_parameter: # 2021/08/23ではTopic数のみの指定
  topic_start: 2
  topic_limit: 40
  topic_step: 1
  ```
