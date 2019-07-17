import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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
TRAIN_SIZE = 100000000

TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'

train_data = read_file(TRAIN_PATH)
test_data = read_file(TEST_PATH)
all_data = pd.concat((train_data,  test_data), axis = 0, ignore_index = True, sort = False)
del train_data
del test_data

# 训练tf_idf和countVectorize 如果内存支持的话，sample越多越好(先从全部的数据中，之后慢慢减少试试)，但是要注意的是sklearn参数的设置，比如保留比例啥的)
# sample_data = all_data.sample(frac=0.1)
sample_data = all_data
text_source = np.hstack([sample_data['query'].values,sample_data['title'].values])


# tf_idf
# 超过1%的词可以作为停用词，出现次数小于2的也不计入(因为这些词只在query或者title中出现)
tf_idf = TfidfVectorizer(max_df=0.01, min_df=2).fit(text_source)
with open('tf_idf_model.pickle', 'wb') as f:
    pickle.dump(tf_idf, f)


# CountVectorizer
cv = CountVectorizer(max_df=0.01, min_df=2).fit(text_source)
with open('cv_model.pickle', 'wb') as f:
    pickle.dump(cv, f)

# word2Vec
text_source = [i.split() for i in text_source]
w2v = Word2Vec(sentences=text_source ,size=256, window=5, min_count=5, negative=0, workers=16)
w2v.wv.save_word2vec_format("word2vec.vector", binary=True)

# fasttext
fasttext = FastText(sentences=text_source ,size=256, window=5, min_count=5, negative=0, workers=16)
fasttext.wv.save_word2vec_format("fast2vec.vector", binary=True)

# docVec
text_source = [TaggedDocument(doc, [i]) for i, doc in enumerate(text_source)]
docvec = Doc2Vec(documents=text_source, dm=1, vector_size=256, window=5, min_count=5, negative=0, workers=16)
docvec.wv.save_word2vec_format("doc2vec.vector", binary=True)
