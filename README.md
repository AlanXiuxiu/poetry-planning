# Poetry Planning for ViVi_3.0

## Introduction

ViVi的Poetry Planning部分，给定句子，可以生成所需数量的关键词，
关键词被限制在一定语料范围内，例如全唐诗、全宋词等

## Prerequsites

- Python3.6
- jieba==0.39
- gensim==2.0.0
- Download modern word2vec model from：
https://github.com/Embedding/Chinese-Word-Vectors
（Thanks for *Shen Li, Zhe Zhao, Renfen Hu, Wensi Li, Tao Liu, Xiaoyong Du, Analogical Reasoning on Chinese Morphological and Semantic Relations, ACL 2018.*）

## Main Process

- `gen_train_data()`

首先初始化segment()分词部分（利用sxhy的词库进行分词）

再利用poems()将raw语料库中的诗词语料处理成规范的格式，放入列表并生成poem.txt

Ranked_words()遵循TextRank算法，对文本poems.txt中的词计数，然后迭代计算，排名

从poems诗中筛选出七言绝句，然后根据wordrank排名选择每一句诗的关键词keyword，

生成用于训练词向量的plan_data.txt以及供查看的plan_history.txt

- `train_planner()`

利用之前plan出的关键词（四个词一组，对应一首诗的四句）plan_data.txt，训练词向量模型

生成plan_model.bin

- `class Planner`

分为extract和expand两部分

extract(text)对输入的句子进行断句，分词，过滤出在ranked_words词表中的词，作为keywords

expand(keywords)在关键词数目不够时被调用：

如果已有的keywords可以在词向量模型中被找到，就在这些词的附近找随机的词向量所对应的词；

如果已有的keywords可以没有出现在词向量模型中，则按照ranked_words的顺序逐一roll，

每个词都有一定概率被选中，这样选取出的词会是在rank中排名较高的词

## 2019/4/9 update

- 添加了现代词向量。首先通过现代词词向量模型寻找相近的词，如果在古代词词表或其embedding中存在该词，即可确定下来

- 对古代词向量模型生成的过程做了优化，在古代词向量中出现次数少于5次的词被过滤掉

- 优化了个别情况下生成关键词数目混乱的bug


