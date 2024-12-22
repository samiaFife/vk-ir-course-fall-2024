#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution"""

import argparse
from collections import Counter
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import tqdm
from pandarallel import pandarallel

pandarallel.initialize(verbose=0)


def preprocess(text):
    import pymorphy3 as pymorphy
    morph = pymorphy.MorphAnalyzer(lang='ru')
    tokens = text.split()
    lemmatized = [morph.parse(token)[0].normal_form for token in tokens]
    return lemmatized


class BM25F:
    def __init__(self, docs, field_weights, k1=1.2, b=0.75):
        self.docs = docs
        self.field_weights = field_weights
        self.k1 = k1
        self.b = b
        self.doc_count = len(docs)
        self.avgdl = self._calculate_avgdl()
        self.idf = {}
        self.tf = []
        self._calculate_idf_and_tf()

    def _calculate_avgdl(self):
        total_length = sum(len(self.docs[field][i]) for i in range(self.doc_count) for field in self.docs.columns)
        return total_length / self.doc_count

    def _calculate_idf_and_tf(self):
        term_doc_freq = Counter()
        for i in range(self.doc_count):
            term_count = Counter()
            for field in self.docs.columns:
                terms = self.docs[field][i]
                term_count.update(terms)
                unique_terms = set(terms)
                for term in unique_terms:
                    term_doc_freq[term] += 1
            self.tf.append(term_count)

        for term, freq in term_doc_freq.items():
            self.idf[term] = 1 + np.log((self.doc_count + 1) / (freq + 1))

    def get_scores(self, query):
        scores = np.zeros(self.doc_count)
        for term in query:
            if term in self.idf:
                for i in range(self.doc_count):
                    score = 0
                    for field in self.docs.columns:
                        f = self.docs[field][i].count(term)
                        field_length = len(self.docs[field][i])
                        score += (self.idf[term] * self.field_weights[field] *
                                  (f * (self.k1 + 1)) /
                                  (f + self.k1 * (1 - self.b + self.b * (field_length / self.avgdl))))
                    scores[i] += score
        return scores


def process_file(path, required_docs):
    chunk_size = 10000
    # total_lines = sum(1 for _ in tqdm.tqdm(open(path, encoding='utf-8'), desc='Counting lines', leave=False))
    # print(total_lines)
    total_lines = 1258656
    total_chunks = total_lines // chunk_size
    df = pd.DataFrame()
    for chunk in tqdm.tqdm(pd.read_csv(path, sep='\t', names=['doc_id', 'url', 'title', 'body'], chunksize=chunk_size),
                           total=total_chunks,
                           desc="Processing file chunks"):
        filtered_chunk = chunk[chunk['doc_id'].isin(required_docs)].copy()
        # filtered_chunk.loc[:, 'text'] = filtered_chunk['title'] + ' ' + filtered_chunk['body']
        df = pd.concat([df, filtered_chunk[['doc_id', 'title', 'body']]], ignore_index=True)
    return df


def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Будем измерять время работы скрипта
    start = timer()

    # Тут вы, скорее всего, должны проинициализировать фиксированным seed'ом генератор случайных чисел для того, чтобы ваше решение было воспроизводимо.
    # Например (зависит от того, какие конкретно либы вы используете):
    #
    # random.seed(42)
    # np.random.seed(42)
    # и т.п.

    # Дальше вы должны:
    # - загрузить датасет VK MARCO из папки args.data_dir
    # - реализовать какой-то из классических алгоритмов текстового ранжирования, например TF-IDF или BM25
    # - при необходимости, подобрать гиперпараметры с использованием трейна или валидации
    # - загрузить пример сабмишна из args.data_dir/sample_submission.csv
    # - применить алгоритм ко всем запросам и документам из примера сабмишна
    # - переранжировать документы из примера сабмишна в соответствии со скорами, которые выдает ваш алгоритм
    # - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file

    sample_submission_df = pd.read_csv(args.data_dir + '/sample_submission.csv', names=['qid', 'doc_id'])
    required_docs = sample_submission_df['doc_id'].tolist()
    qd_rank = sample_submission_df.groupby('qid')['doc_id'].apply(list).to_dict()
    # docs = process_file(args.data_dir + '/vkmarco-docs.tsv', required_docs)
    # docs.to_csv(args.data_dir + '/DOCS_TMP.csv', index=False)
    docs = pd.read_csv(args.data_dir + '/DOCS_TMP.csv')

    queries = pd.read_csv(args.data_dir + '/vkmarco-doceval-queries.tsv', sep='\t', names=['qid', 'qtext'])
    res = []
    with tqdm.tqdm(total=len(queries), desc='Ranking queries') as pbar:
        for _, query in queries.iterrows():
            qid = query['qid']
            qtext = query['qtext']
            relevant_docs = qd_rank.get(str(qid))
            relevant_doc_texts = docs.loc[docs['doc_id'].isin(relevant_docs), ['title', 'body']].fillna('')
            tokenized_docs = relevant_doc_texts.parallel_apply(lambda row: {
                'title': preprocess(row['title']),
                'body': preprocess(row['body'])
            }, axis=1)
            tokenized_docs = pd.DataFrame(tokenized_docs.tolist(), columns=['title', 'body'])
            bm25f = BM25F(tokenized_docs, {'title': 2.0, 'body': 1.0})
            scores = bm25f.get_scores(qtext)
            relevant_scores = pd.DataFrame({
                'doc_id': relevant_docs,
                'score': scores
            })
            ranked_docs = relevant_scores.sort_values(by='score', ascending=False)
            for _, row in ranked_docs.iterrows():
                res.append({'QueryId': qid, 'DocumentId': row['doc_id']})
            pbar.update(1)
    pd.DataFrame(res).to_csv(args.submission_file, index=False)
    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
