# <div style="text-align: center;">LDA Sample Tools</div>

Latent Dirichlet Allocation（潜在的ディリクレ配分法:LDA）と呼ばれるトピックモデルのサンプルプログラムを作成  

## TODO

- [ ] 形態素解析器
  - [ ] Janome
    - [X] ~~*[文, 文]のリストに対応*~~ [2021-08-10]
    - [X] ~~*文に対応*~~ [2021-08-11]
    - [ ] 品詞フィルターの作成  
  - [ ] 柔軟に形態素解析器を変える 
- [ ] 前処理
  - [X] ~~*コーパス作成*~~ [2021-08-15]
  - [X] ~~*word-id辞書作成*~~ [2021-08-15]
  - [ ] 単語の出現頻度分析
- [ ] LDA
  - [ ] model作成
    - [ ] パラメーター
    - [X] ~~*マルチコア対応*~~ [2021-08-16]
    - [X] ~~*モデル評価*~~ [2021-08-19]
      - [X] ~~*avg_topic_coherence*~~ [2021-08-19]
  - [ ] model保存
  - [X] ~~*topic wordの抽出*~~ [2021-08-19]
  - [ ] 文章に対してのTopic分析

## 構成
|          |               | 
| -------- | :------------ | 
| python   | 3.6.9         | 
| lda      | gensim 4.0.1  | 
| analyzer | Janome 0.3.10 | 

## 実行
2021/08/19現在
```
docker-compose build
docker-compose up -d
docker-compose run lda-tool python lda_model.py
```