import pandas as pd
import numpy as np
import random
from helper_fun import lcsubstr_lens, lcseque_lens, lev, levRatio, euclidean, pearson, Jaccrad, common_word_ratio_adjust_with_tfidf, cos_sim
import scipy.special as special
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing import sequence
from scipy import sparse
import pickle

def read_file(path, is_test = False):
    data = pd.read_csv(path, sep=',', header=None)
    if(data.shape[1] == 5):
        data.columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
    elif(data.shape[1] == 4):
        data.columns = ['query_id', 'query', 'query_title_id', 'title']
        data['label'] = -1
    return data


# TRAIN_PATH = '/home/kesci/input/bytedance/first-round/train.csv'
# TEST_PATH = '/home/kesci/input/bytedance/first-round/test.csv'
# UNIQUE_PATH = '/home/kesci/work/nunique.csv'
# CONVERT_QUERY_PATH = '/home/kesci/work/convert_query.csv'
# CONVERT_TITLE_PATH = '/home/kesci/work/convert_title.csv'
# CV_PATH = '/home/kesci/work/all_cv.npz'
TRAIN_SIZE = 20000

TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
#UNIQUE_PATH = 'nunique.csv'
#CONVERT_QUERY_PATH = 'convert_query.csv'
#CONVERT_TITLE_PATH = 'convert_title.csv'


TRAIN_QUERY_PATH = 'features/train_query.pickle'
TRAIN_TITLE_PATH = 'features/train_title.pickle'
TRAIN_Y_PATH = 'features/train_y.pickle'
TEST_QUERY_PATH = 'features/test_query.pickle'
TEST_TITLE_PATH = 'features/test_title.pickle'

TF_IDF_TRAIN_PATH = 'features/train_tf_idf.npz'
TF_IDF_TEST_PATH = 'features/test_tf_idf.npz'
CV_TRAIN_PATH = 'features/train_countvectorizer.npz'
CV_TEST_PATH = 'features/test_countvectorizer.npz'

WV_EMBEDDING_PATH1 = "features/word2vec_embedding1.pickle"
WV_EMBEDDING_PATH2 = "features/word2vec_embedding2.pickle"
FT_EMBEDDING_PATH1 = "features/fasttext_embedding1.pickle"
FT_EMBEDDING_PATH2 = "features/fasttext_embedding2.pickle"
DV_EMBEDDING_PATH1 = "features/doc2vec_embedding1.pickle"
DV_EMBEDDING_PATH2 = "features/doc2vec_embedding2.pickle"

TF_IDF_MODEL_PATH = "helper_model/tf_idf.model"
CV_MODEL_PATH = "helper_model/countVectorizer.model"
WV_MODEL_PATH = "helper_model/word2vec.model"
FT_MODEL_PATH = "helper_model/fasttext.model"
DV_MODEL_PATH = "helper_model/doc2vec.model"

TRAIN_DISCRETE_FEATURES_PATH = "features/train_discrete_features.csv"
TEST_DISCRETE_FEATURES_PATH = "features/test_discrete_features.csv"

train_data = read_file(TRAIN_PATH)
test_data = read_file(TEST_PATH)
all_data = pd.concat((train_data,  test_data), axis = 0, ignore_index = True, sort = False)
del train_data
del test_data

def list_generator(row):
    return row.replace('\n', '').split(' ')

def length_generator(row):
    return len(row.replace('\n', '').split(' '))

def isin_ratio(row):
    return np.mean([1 if w in row['title_list'] else 0 for w in row['query_list']])

def isin(row):
    return 1 if row['query'] in row['title'] else 0

all_data['query_list'] = all_data['query'].apply(list_generator)
all_data['title_list'] = all_data['title'].apply(list_generator)
all_data['label'] = all_data['label'].astype('int8')

# ----- nunique 特征 -----（全局特征)
# all_data['query_nunique_title'] = all_data.groupby('query').title.transform('nunique')
# all_data['title_nunique_query'] = all_data.groupby('title').query.transform('nunique')

# ----- 长度特征 ------（局部特征）
all_data['query_length'] = all_data['query_list'].apply(lambda x : len(x))
all_data['title_length'] = all_data['title_list'].apply(lambda x : len(x))
all_data['query_length_chu_title_length'] = all_data['query_length'] / all_data['title_length']
all_data['query_length_jian_title_length'] = all_data['query_length'] - all_data['title_length']
all_data['query_length_ration_title_length'] = np.abs(all_data['query_length_jian_title_length']) / np.max(all_data[['query_length', 'title_length']], axis=1)

# ------ 是否存在 ------(局部特征）
all_data['query_isin_title'] = all_data[['query', 'title']].apply(isin, axis=1)
all_data['query_isin_ratio_title'] = all_data[['query_list', 'title_list']].apply(isin_ratio, axis=1)

# ------ 公共词特征 ------(局部特征)
query_word_set = all_data['query'].apply(lambda x: x.split(' ')).apply(set).values
title_word_set = all_data['title'].apply(lambda x: x.split(' ')).apply(set).values
all_data['common_number']  = [len(query_word_set[i] & title_word_set[i]) for i in range(len(query_word_set))]
all_data['common_number_ratio']  = [len(query_word_set[i] & title_word_set[i]) / max(len(query_word_set[i]), len(title_word_set[i])) for i in range(len(query_word_set))]

# ------ 文本相似度特征 ------(局部特征)
all_data['prefix_title_levenshtein'] = all_data[['query_list', 'title_list']].apply(lambda x: lev(x['query_list'], x['title_list']), axis=1)
all_data['prefix_title_levenshtein_ratio'] = all_data[['query_list', 'title_list']].apply(lambda x: levRatio(x['query_list'], x['title_list']), axis=1)
all_data['prefix_title_jaccard'] = all_data[['query_list', 'title_list']].apply(lambda x: Jaccrad(x['query_list'], x['title_list']), axis=1)
# all_data['prefix_title_pearson'] = all_data[['query_list', 'title_list']].apply(lambda x: pearson(x['query_list'], x['title_list']), axis=1)
all_data['prefix_title_lcsubstr_lens'] = all_data[['query_list', 'title_list']].apply(lambda x: lcsubstr_lens(x['query_list'], x['title_list']), axis=1)
all_data['prefix_title_lcseque_lens'] = all_data[['query_list', 'title_list']].apply(lambda x: lcseque_lens(x['query_list'], x['title_list']), axis=1)

# ------ tf-idf特征 ------
with open(TF_IDF_MODEL_PATH, mode="rb") as f:
    tf_idf = pickle.load(f)
tf_idf_query = tf_idf.transform(all_data['query'])
tf_idf_title = tf_idf.transform(all_data['title'])

all_data['common_word_ratio_adjust_with_tfidf'] = common_word_ratio_adjust_with_tfidf(all_data, tf_idf.vocabulary_, tf_idf_query, tf_idf_title)
all_data['tf_idf_simi_cos'] = cos_sim(tf_idf_query, tf_idf_title)
all_data['tf_idf_simi_l2'] = euclidean(tf_idf_query, tf_idf_title)

# 保存tf_idf特征
train_tf_idf = sparse.hstack((tf_idf_query[:TRAIN_SIZE], tf_idf_title[:TRAIN_SIZE]))
sparse.save_npz(TF_IDF_TRAIN_PATH, train_tf_idf)
test_tf_idf = sparse.hstack((tf_idf_query[TRAIN_SIZE:], tf_idf_title[TRAIN_SIZE:]))
sparse.save_npz(TF_IDF_TEST_PATH, test_tf_idf)

# ------ countVecotor特征 -----
with open(CV_MODEL_PATH, mode="rb") as f:
    cv = pickle.load(f)
cv_query = cv.transform(all_data['query'])
cv_title = cv.transform(all_data['title'])

all_data['cv_simi_cos'] = cos_sim(cv_query, cv_title)
all_data['cv_simi_l2'] = euclidean(cv_query, cv_title)

# 保存cv特征
train_cv = sparse.hstack((cv_query[:TRAIN_SIZE], cv_title[:TRAIN_SIZE]))
sparse.save_npz(CV_TRAIN_PATH, train_cv)
test_cv = sparse.hstack((cv_query[TRAIN_SIZE:], cv_title[TRAIN_SIZE:]))
sparse.save_npz(CV_TEST_PATH, test_cv)

# 保存训练集X，Y
Query_train = sequence.pad_sequences(all_data['query_list'][:TRAIN_SIZE], maxlen=10, padding='post', truncating='post')
Title_train = sequence.pad_sequences(all_data['title_list'][:TRAIN_SIZE], maxlen=20, padding='post',truncating='post')
Y_train = all_data['label'][:TRAIN_SIZE]

Query_test = sequence.pad_sequences(all_data['query_list'][TRAIN_SIZE:], maxlen=10, padding='post', truncating='post')
Title_test = sequence.pad_sequences(all_data['title_list'][TRAIN_SIZE:], maxlen=20, padding='post',truncating='post')

with open(TRAIN_QUERY_PATH, mode="wb") as f:
    pickle.dump(Query_train, f)
with open(TRAIN_TITLE_PATH, mode="wb") as f:
    pickle.dump(Title_train, f)
with open(TRAIN_Y_PATH, mode="wb") as f:
    pickle.dump(Y_train, f)
with open(TEST_QUERY_PATH, mode="wb") as f:
    pickle.dump(Query_test, f)
with open(TEST_TITLE_PATH, mode="wb") as f:
    pickle.dump(Title_test, f)


# ----- word2vec ------
wordvec_model = KeyedVectors.load_word2vec_format(WV_MODEL_PATH, binary=True, encoding='utf8', unicode_errors='ignore')
n = 2000000 # 这边具体多少得取决于训练集中的最大word的index
chunk_size = 1000000
w2_embeddings_matrix = np.zeros((n, wordvec_model.vector_size))
keys = wordvec_model.wv.vocab.keys()
for word in range(n):
    if str(word) in keys:
        w2_embeddings_matrix[word] = wordvec_model[str(word)]
all_data['w2v_simi'] = all_data[['query_list','title_list']].apply(lambda x: wordvec_model.wmdistance(x['query_list'], x['title_list']), axis=1)
mean = np.mean([x for x in all_data['w2v_simi'] if x != float("inf")])
all_data['w2v_simi'] = all_data['w2v_simi'].apply(lambda x: mean if x == float("inf") else x)
with open(WV_EMBEDDING_PATH1, mode="wb") as f:
    pickle.dump(w2_embeddings_matrix[:chunk_size], f)
with open(WV_EMBEDDING_PATH2, mode="wb") as f:
    pickle.dump(w2_embeddings_matrix[chunk_size:], f)


# ----- fasttext ------
fastvec_model = KeyedVectors.load_word2vec_format(FT_MODEL_PATH, binary=True, encoding='utf8',unicode_errors='ignore')
ft_embeddings_matrix = np.zeros((n, fastvec_model.vector_size))
keys = fastvec_model.wv.vocab.keys()
for word in range(n):
    if str(word) in keys:
        ft_embeddings_matrix[word] = fastvec_model[str(word)]
all_data['ft_simi'] = all_data[['query_list','title_list']].apply(lambda x: fastvec_model.wmdistance(x['query_list'], x['title_list']), axis=1)
mean = np.mean([x for x in all_data['ft_simi'] if x != float("inf")])
all_data['ft_simi'] = all_data['ft_simi'].apply(lambda x: mean if x == float("inf") else x)
with open(FT_EMBEDDING_PATH1, mode="wb") as f:
    pickle.dump(ft_embeddings_matrix[:chunk_size], f)
with open(FT_EMBEDDING_PATH2, mode="wb") as f:
    pickle.dump(ft_embeddings_matrix[chunk_size:], f)

# ----- doc2vec -----
docvec_model = KeyedVectors.load_word2vec_format(DV_MODEL_PATH, binary=True, encoding='utf8',unicode_errors='ignore')
dv_embeddings_matrix = np.zeros((n, docvec_model.vector_size))
keys = docvec_model.wv.vocab.keys()
for word in range(n):
    if str(word) in keys:
        dv_embeddings_matrix[word] = docvec_model[str(word)]
all_data['dv_simi'] = all_data[['query_list','title_list']].apply(lambda x: docvec_model.wmdistance(x['query_list'], x['title_list']), axis=1)
mean = np.mean([x for x in all_data['dv_simi'] if x != float("inf")])
all_data['dv_simi'] = all_data['dv_simi'].apply(lambda x: mean if x == float("inf") else x)
with open(DV_EMBEDDING_PATH1, mode="wb") as f:
    pickle.dump(dv_embeddings_matrix[:chunk_size], f)
with open(DV_EMBEDDING_PATH2, mode="wb") as f:
    pickle.dump(dv_embeddings_matrix[chunk_size:], f)

all_data = all_data[['query_nunique_title', 'title_nunique_query','query_length', 'title_length', 'query_length_chu_title_length','query_length_jian_title_length','query_length_ration_title_length','query_isin_title', 'query_isin_ratio_title', 'common_number','common_number_ratio', 'prefix_title_levenshtein','prefix_title_levenshtein_ratio', 'prefix_title_jaccard','prefix_title_lcsubstr_lens', 'prefix_title_lcseque_lens','common_word_ratio_adjust_with_tfidf', 'tf_idf_simi_cos','tf_idf_simi_l2', 'cv_simi_cos', 'cv_simi_l2', 'w2v_simi', 'ft_simi','dv_simi']]

all_data[:TRAIN_SIZE].to_csv(TRAIN_DISCRETE_FEATURES_PATH)
all_data[TRAIN_SIZE:].to_csv(TEST_DISCRETE_FEATURES_PATH)