from scipy.stats import pearsonr,spearmanr,kendalltau
import numpy as np
def lcsubstr_lens(s1, s2):
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0   #最长匹配的长度
    p=0  #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return mmax

def lcseque_lens(s1, s2):
     # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]
    # d用来记录转移方向
    d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:            #字符匹配成功，则该位置的值为左上方的值加1
                m[p1+1][p2+1] = m[p1][p2]+1
                d[p1+1][p2+1] = 'ok'
            elif m[p1+1][p2] > m[p1][p2+1]:  #左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1+1][p2+1] = m[p1+1][p2]
                d[p1+1][p2+1] = 'left'
            else:                           #上值大于左值，则该位置的值为上值，并标记方向up
                m[p1+1][p2+1] = m[p1][p2+1]
                d[p1+1][p2+1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    s = []
    while m[p1][p2]:    #不为None时
        c = d[p1][p2]
        if c == 'ok':   #匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1-1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  #根据标记，向左找下一个
            p2 -= 1
        if c == 'up':   #根据标记，向上找下一个
            p1 -= 1
    return len(s)

def compute_convert(pos, sums, label):
    if np.isnan(sums): # not oppear in train
        return -1
    if label != -1 and sums == 1: # only oppear once
        return -1
    if label == 1:
        return (pos - 1) / (sums - 1)
    elif label == 0:
        return pos / (sums - 1)
    else:
        return pos / sums

def find_longest_prefix(str_list):
    if not str_list:
        return ''
    str_list.sort(key = lambda x: len(x))
    shortest_str = str_list[0]
    max_prefix = len(shortest_str)
    flag = 0
    for i in range(max_prefix):
        for one_str in str_list:
            if one_str[i] != shortest_str[i]:
                return shortest_str[:i]
                break
    return shortest_str


def euclidean(vector_a, vector_b):
    # 如果两数据集数目不同，计算两者之间都对应有的数
    vector_c = np.abs(vector_a - vector_b)
    c_c = np.sqrt((vector_c * vector_c.T).todense().diagonal())
    return c_c.T


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    a_b = (vector_a * vector_b.T).todense().diagonal()
    a_a = np.sqrt((vector_a * vector_a.T).todense().diagonal())
    b_b = np.sqrt((vector_b * vector_b.T).todense().diagonal())
    cos = a_b / np.multiply(a_a, b_b)
    sim = 0.5 + 0.5 * cos
    return sim.T

def pearson(p, q):
    p = [int(i) for i in p]
    q = [int(i) for i in q]
    # 只计算两者共同有的
    same = 0
    for i in p:
        if i in q:
            same += 1

    n = same
    # 分别求p，q的和
    sumx = sum([p[i] for i in range(n)])
    sumy = sum([q[i] for i in range(n)])
    # 分别求出p，q的平方和
    sumxsq = sum([p[i] ** 2 for i in range(n)])
    sumysq = sum([q[i] ** 2 for i in range(n)])
    # 求出p，q的乘积和
    sumxy = sum([p[i] * q[i] for i in range(n)])
    # print sumxy
    # 求出pearson相关系数
    up = sumxy - sumx * sumy / n
    down = ((sumxsq - pow(sumxsq, 2) / n) * (sumysq - pow(sumysq, 2) / n)) ** .5
    # 若down为零则不能计算，return 0
    if down == 0: return 0
    r = up / down
    return r


def manhattan(p,q):
#只计算两者共同有的
    same = 0
    for i in p:
        if i in q:
            same += 1
#计算曼哈顿距离
    n = same
    vals = range(n)
    distance = sum(abs(int(p[i]) - int(q[i])) for i in vals)
    return distance



def Jaccrad(str1, str2):  # terms_reference为源句子，terms_model为候选句子
    grams_reference = set(str1)  # 去重；如果不需要就改为list
    grams_model = set(str2)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = float(temp / fenmu)  # 交集
    return jaccard_coefficient


# Levenshtein Distance
def lev(str1, str2):
    """Make a Levenshtein Distances Matrix"""
    n1, n2 = len(str1), len(str2)
    lev_matrix = [ [ 0 for i1 in range(n1 + 1) ] for i2 in range(n2 + 1) ]
    for i1 in range(1, n1 + 1):
        lev_matrix[0][i1] = i1
    for i2 in range(1, n2 + 1):
        lev_matrix[i2][0] = i2
    for i2 in range(1, n2 + 1):
        for i1 in range(1, n1 + 1):
            cost = 0 if str1[i1-1] == str2[i2-1] else 1
            elem = min( lev_matrix[i2-1][i1] + 1,
                        lev_matrix[i2][i1-1] + 1,
                        lev_matrix[i2-1][i1-1] + cost )
            lev_matrix[i2][i1] = elem
    return lev_matrix[-1][-1]




def levRatio(str1, str2):
    n1, n2 = len(str1), len(str2)
    lev_matrix = [ [ 0 for i1 in range(n1 + 1) ] for i2 in range(n2 + 1) ]
    for i1 in range(1, n1 + 1):
        lev_matrix[0][i1] = i1
    for i2 in range(1, n2 + 1):
        lev_matrix[i2][0] = i2
    for i2 in range(1, n2 + 1):
        for i1 in range(1, n1 + 1):
            cost = 0 if str1[i1-1] == str2[i2-1] else 1
            elem = min( lev_matrix[i2-1][i1] + 1,
                        lev_matrix[i2][i1-1] + 1,
                        lev_matrix[i2-1][i1-1] + cost )
            lev_matrix[i2][i1] = elem
    total_length = str1.__len__() + str2.__len__()
    return (total_length - lev_matrix[-1][-1]) / total_length

'''
用tfidf作为系数，调整共现词比例
'''
def common_word_ratio_adjust_with_tfidf(merge_data, word_to_index, q1_tfidf, q2_tfidf):
    merge = merge_data[['query', 'title']]
    merge.columns = ['q1', 'q2']

    adjusted_common_word_ratio = []

    for i in range(q1_tfidf.shape[0]):
        q1words = {}
        q2words = {}
        for word in merge.loc[i, 'q1'].split():
            q1words[word] = q1words.get(word, 0) + 1
        for word in merge.loc[i, 'q2'].split():
            q2words[word] = q2words.get(word, 0) + 1

        sum_shared_word_in_q1 = sum([q1words[w] * q1_tfidf[i, word_to_index[w]] for w in q1words if w in q2words and w in word_to_index])
        sum_shared_word_in_q2 = sum([q2words[w] * q2_tfidf[i, word_to_index[w]] for w in q2words if w in q1words and w in word_to_index])
        sum_tol = sum(q1words[w] * q1_tfidf[i, word_to_index[w]] for w in q1words if w in word_to_index) + sum(q2words[w] * q2_tfidf[i, word_to_index[w]] for w in q2words if w in word_to_index)
        if 1e-6 > sum_tol:
            adjusted_common_word_ratio.append(0.)
        else:
            adjusted_common_word_ratio.append(1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol)

    return adjusted_common_word_ratio