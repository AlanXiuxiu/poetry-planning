# -*- coding:utf-8 -*-

import os
from random import random, shuffle

import jieba
from gensim import models

from char_dict import end_of_sentence, start_of_sentence
from data_utils import split_sentences, NUM_OF_SENTENCES
from paths import plan_history_path, plan_data_path, check_uptodate
from paths import save_dir
from poems import Poems
from rank_words import RankedWords
from segment import Segmenter

_plan_model_path = os.path.join(save_dir, 'plan_model.bin')


def gen_train_data():
    print("Generating training data ...")
    segmenter = Segmenter()
    poems = Poems()
    poems.shuffle()
    ranked_words = RankedWords()
    plan_data = []  # 从poem.txt中把筛选出来的诗做关键词提取，按wordrank分优先级
    plan_history = []  # 记录每句诗以及从诗中提取的planning word，做参照查看
    for poem in poems:
        if len(poem) != 4:
            continue  # Only consider quatrains.
        valid = True
        context = start_of_sentence()
        gen_lines = []
        keywords = []
        for sentence in poem:
            if len(sentence) != 7:
                valid = False
                break
            words = list(filter(lambda seg: seg in ranked_words,
                                segmenter.segment(sentence)))
            if len(words) == 0:
                valid = False
                break
            keyword = words[0]
            for word in words[1:]:
                if ranked_words.get_rank(word) < ranked_words.get_rank(keyword):
                    keyword = word
            gen_line = sentence + end_of_sentence() + \
                       '\t' + keyword + '\t' + context + '\n'
            gen_lines.append(gen_line)
            keywords.append(keyword)
            context += sentence + end_of_sentence()
        if valid:
            plan_data.append('\t'.join(keywords) + '\n')
            plan_history.extend(gen_lines)
    with open(plan_data_path, 'w', encoding='UTF-8') as fout:
        for line in plan_data:
            fout.write(line)
    with open(plan_history_path, 'w', encoding='UTF-8') as fout:
        for line in plan_history:
            fout.write(line)


def train_planner():
    print("Training Word2Vec-based planner ...")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not check_uptodate(plan_data_path):
        gen_train_data()
    word_lists = []
    with open(plan_data_path, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            word_lists.append(line.strip().split('\t'))
    model = models.Word2Vec(word_lists, size=512, min_count=5)
    model.save(_plan_model_path)


class Planner:

    def __init__(self):
        self.ranked_words = RankedWords()
        if not os.path.exists(_plan_model_path):
            train_planner()
        self.model = models.Word2Vec.load(_plan_model_path)

    def plan(self, text):
        return self._expand(self._extract(text))

    def _extract(self, text):
        def extract_from_sentence(sentence):
            return filter(lambda w: w in self.ranked_words,
                          jieba.lcut(sentence))

        keywords = set()
        for sentence in split_sentences(text):
            keywords.update(extract_from_sentence(sentence))
        return keywords

    def _expand(self, keywords):
        if len(keywords) < NUM_OF_SENTENCES:
            filtered_keywords = list(filter(lambda w: w in \
                                                      self.model.wv, keywords))
            if len(filtered_keywords) > 0:
                similars = self.model.wv.most_similar(
                    positive=filtered_keywords)
                # Sort similar words in decreasing similarity with randomness.
                similars = sorted(similars, key=lambda x: x[1] * random())
                for similar in similars:
                    keywords.add(similar[0])
                    if len(keywords) == NUM_OF_SENTENCES:
                        break
            prob_sum = sum(1. / (i + 1) \
                           for i, word in enumerate(self.ranked_words) \
                           if word not in keywords)
            rand_val = prob_sum * random()
            word_idx = 0
            s = 0
            while len(keywords) < NUM_OF_SENTENCES \
                    and word_idx < len(self.ranked_words):
                word = self.ranked_words[word_idx]
                s += 1.0 / (word_idx + 1)
                if word not in keywords and rand_val < s:
                    keywords.add(word)
                word_idx += 1
        results = list(keywords)
        shuffle(results)
        return results


# For testing purpose.
if __name__ == '__main__':
    planner = Planner()
    inputs = ["春天到了，桃花开了。",
              "举杯饮酒，思乡情怯",
              "牧童遥指杏花村",
              "中秋节的夜晚，想起故乡的月亮"
              ]
    for input in inputs:
        keywords = planner.plan(input)
        print('{0:{2}^12}  ->  {1}\t'.format(input, keywords, chr(12288)))
