import string
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import Word2Vec


def remove_singleton_words(list_of_lines: list[list[str]]):
    freq_d = {}
    for line in list_of_lines:
        for token in line:
            freq_d[token] += 1

    out_lines = [
        [token for token in line if freq_d[token] > 1]
        for line in list_of_lines
    ]

    return out_lines


def get_top_cosine_similarities(target_word: str, word2vec_results: pd.DataFrame, corpus_words: list[str], num_display=10):
    """
    :param target_word: Name of the word (whose vector representation is stored in word2vec_results) to analyze cosine
    similarity against the other words in corpus
    :param word2vec_results: DataFrame storing vector representations of the words in corpus
    :param corpus_words: The words in corpus with index order equal to their entry order as vectors in word2vec output
    :param num_display: (Optional) Total number of words to output having the greatest magnitude of cosine similarity to target_word
    :return: Series with num_display words having the greatest magnitude of cosine similarity to target_word along with
     target_word itself
    """
    cos_scores = {}
    for word in corpus_words:
        cos_scores[word] = cosine_similarity(word2vec_results[target_word], word2vec_results[word])
    cos_results_s = pd.Series(data=cos_scores)
    num_display = num_display // 2
    _out_s = cos_results_s.sort_values(ascending=False).head(num_display+1)
    _out_s = pd.concat([_out_s, cos_results_s.sort_values().head(num_display)])

    return _out_s


def cosine_similarity(a, b):
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    score = numerator / denominator

    return score


root_dir = r'C:\Users\nikhi\Dropbox\JaldiKaro\DataScience'

punc_d = {p: '' for p in list(string.punctuation)}
no_punc_tbl = str.maketrans(punc_d)
docu_lines = []
with open(rf'{root_dir}/NLP/src/Why I am Not a Liberal - D Brooks - NYT.txt', 'r', encoding='utf-8') as input_file:
    input_line = input_file.readline()
    while input_line != '':
        docu_lines.append(input_line)
        input_line = input_file.readline()

docu_lines = [w.strip('\n').translate(no_punc_tbl) for w in docu_lines if w != '\n']
processed_lines = [
    [word for word in doc_line.lower().split()]
    for doc_line in docu_lines
]

# dictionary = corpora.Dictionary(processed_lines)
# dictionary.save(f'{root_dir}/NLP/src/dbrooks_part_one.dict')
corpus = corpora.Dictionary.load(f'{root_dir}/NLP/src/dbrooks_part_one.dict')
model = Word2Vec(sentences=processed_lines, vector_size=100, window=5, min_count=1, workers=6)

word_count_s = pd.Series(corpus.token2id)
word_idx_list = model.wv.index_to_key
model_training_results = pd.DataFrame(model.wv.vectors.T, columns=word_idx_list)

print(get_top_cosine_similarities('lefties', model_training_results, word_idx_list))
