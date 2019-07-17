import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, BatchNormalization, concatenate, Subtract, Dot, Multiply, Bidirectional, Lambda
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.initializers import glorot_uniform
from keras.layers.noise import GaussianNoise
from keras import backend as K
from keras import optimizers
import tensorflow as tf
import keras.callbacks as kcallbacks
import pickle

TRAIN_QUERY_PATH = '../features/train_query.pickle'
TRAIN_TITLE_PATH = '../features/train_title.pickle'
TRAIN_Y_PATH = '../features/train_y.pickle'
TEST_QUERY_PATH = '../features/test_query.pickle'
TEST_TITLE_PATH = '../features/test_title.pickle'

TF_IDF_TRAIN_PATH = '../features/train_tf_idf.npz'
TF_IDF_TEST_PATH = '../features/test_tf_idf.npz'
CV_TRAIN_PATH = '../features/train_countvectorizer.npz'
CV_TEST_PATH = '../features/test_countvectorizer.npz'

WV_EMBEDDING_PATH1 = "../features/word2vec_embedding1.pickle"
WV_EMBEDDING_PATH2 = "../features/word2vec_embedding2.pickle"
FT_EMBEDDING_PATH = "../features/fasttext_embedding.pickle"
DV_EMBEDDING_PATH = "../features/doc2vec_embedding.pickle"

TRAIN_DISCRETE_FEATURES_PATH = "../features/train_discrete_features.csv"
TEST_DISCRETE_FEATURES_PATH = "../features/test_discrete_features.csv"

MAX_SEQUENCE_LENGTH1 = 10  # 20 for character level and 15 for word level
MAX_SEQUENCE_LENGTH2 = 20
EMBEDDING_DIM = 300


with open(TRAIN_QUERY_PATH, mode="rb") as f:
    train_q1 = pickle.load(f)
with open(TRAIN_TITLE_PATH, mode="rb") as f:
    train_q2 = pickle.load(f)
with open(TRAIN_Y_PATH, mode="rb") as f:
    train_label = pickle.load(f)
with open(WV_EMBEDDING_PATH1, mode="rb") as f:
    embed_matrix1 = pickle.load(f)
with open(WV_EMBEDDING_PATH2, mode="rb") as f:
    embed_matrix2 = pickle.load(f)
embed_matrix = np.vstack([embed_matrix1, embed_matrix2])
# 读取数据
print('train_q1: ', train_q1.shape)
print('train_q2: ', train_q2.shape)
print('train_label: ', train_label.shape)
print('embed_matrix: ', embed_matrix.shape)

with open(TEST_QUERY_PATH, mode="rb") as f:
    test_q1 = pickle.load(f)
with open(TEST_TITLE_PATH, mode="rb") as f:
    test_q2 = pickle.load(f)
# 加载test 数据
print('test_q1: ', test_q1.shape)
print('test_q2: ', test_q2.shape)

# 读取手工特征
train_features = pd.read_csv(TRAIN_DISCRETE_FEATURES_PATH)
test_features = pd.read_csv(TEST_DISCRETE_FEATURES_PATH)

pick_columns = ['query_nunique_title', 'title_nunique_query',
       'query_length', 'title_length', 'query_length_chu_title_length',
       'query_length_jian_title_length', 'query_length_ration_title_length',
       'query_isin_title', 'query_isin_ratio_title', 'common_number',
       'common_number_ratio', 'prefix_title_levenshtein',
       'prefix_title_levenshtein_ratio', 'prefix_title_jaccard',
       'prefix_title_lcsubstr_lens', 'prefix_title_lcseque_lens',
       'common_word_ratio_adjust_with_tfidf', 'tf_idf_simi_cos',
       'tf_idf_simi_l2', 'cv_simi_cos', 'cv_simi_l2', 'w2v_simi', 'ft_simi',
       'dv_simi']


train_features = train_features[pick_columns]
test_features = test_features[pick_columns]
# train_features = (train_features - train_features.min()) / (train_features.max() - train_features.min())
# test_features = (test_features - test_features.min()) / (test_features.max() - test_features.min())
train_features.info()


def trainLSTM(train_q1, train_q2, train_label, embed_matrix, train_features):
    lstm_num = 75
    lstm_drop = 0.3
    BATCH_SIZE = 64  # 128

    q1_train = train_q1
    q2_train = train_q2
    y_train = train_label
    f_train = train_features

    # Define the model
    question1 = Input(shape=(MAX_SEQUENCE_LENGTH1,))
    question2 = Input(shape=(MAX_SEQUENCE_LENGTH2,))

    embed_layer1 = Embedding(embed_matrix.shape[0], EMBEDDING_DIM, weights=[embed_matrix],
                            input_length=MAX_SEQUENCE_LENGTH1, trainable=False)
    embed_layer2 = Embedding(embed_matrix.shape[0], EMBEDDING_DIM, weights=[embed_matrix],
                            input_length=MAX_SEQUENCE_LENGTH2, trainable=False)

    q1_embed = embed_layer1(question1)
    q2_embed = embed_layer2(question2)

    shared_lstm_1 = LSTM(lstm_num, return_sequences=True)
    shared_lstm_2 = LSTM(lstm_num)

    q1 = shared_lstm_1(q1_embed)
    q1 = Dropout(lstm_drop)(q1)
    q1 = BatchNormalization()(q1)
    q1 = shared_lstm_2(q1)
    # q1 = Dropout(0.5)(q1)

    q2 = shared_lstm_1(q2_embed)
    q2 = Dropout(lstm_drop)(q2)
    q2 = BatchNormalization()(q2)
    q2 = shared_lstm_2(q2)
    # q2 = Dropout(0.5)(q2)   # of shape (batch_size, 128)

    # 求distance (batch_size,1)
    d = Subtract()([q1, q2])
    # distance = Dot(axes=1, normalize=False)([d, d])
    # distance = Lambda(lambda x: K.abs(x))(d)
    distance = Multiply()([d, d])
    # 求angle (batch_size,1)
    # angle = Dot(axes=1, normalize=False)([q1, q2])
    angle = Multiply()([q1, q2])
    # merged = concatenate([distance,angle])

    # magic featurues
    magic_input = Input(shape=(train_features.shape[1],))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)
    magic_dense = Dropout(0.3)(magic_dense)

    merged = concatenate([distance, angle, magic_dense])
    merged = Dropout(0.3)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(256, activation='relu')(merged)  # 64
    merged = Dropout(0.3)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(64, activation='relu')(merged)  # 64
    merged = Dropout(0.3)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2, magic_input], outputs=is_duplicate)
    loss, metrics = 'binary_crossentropy', ["accuracy"]

    model.compile(loss=loss, optimizer=Adam(lr=3e-4, beta_1=0.8), metrics=metrics)

    model.summary()

    # define save model
    best_weights_filepath = '../models_save/siamese_lstm_with_magics.hdf5'
    earlyStopping = kcallbacks.EarlyStopping(monitor='val_loss', patience=44, verbose=1, mode='auto')
    saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1,save_best_only=True, mode='auto')

    hist = model.fit([q1_train, q2_train, f_train],
                     y_train,
                     epochs=30,
                     batch_size=BATCH_SIZE,
                     class_weight={0: 1 / np.mean(y_train), 1: 1 / (1 - np.mean(y_train))},
                     #shuffle=True,
                     #callbacks=[earlyStopping, saveBestModel],
                     verbose=1)




# run the model and predict
import time

start = time.time()

trainLSTM(train_q1, train_q2, train_label, embed_matrix, train_features.values)

end = time.time()
print('Training time {0:.3f} 分钟'.format((end - start) / 60))