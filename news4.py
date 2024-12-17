import numpy as np
import pandas as pd
import pickle

from faiss import Index
from tqdm import tqdm
import gc, os
import logging
import time
import lightgbm as lgb
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
import warnings

from news.news3 import all_click_df

warnings.filterwarnings('ignore')


# 节省内存的一个函数
# 减少内存
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                    100 * (
                                                                                                            start_mem - end_mem) / start_mem,
                                                                                                    (
                                                                                                            time.time() - starttime) / 60))
    return df


data_path = '../data_raw/'
save_path = '../temp_results/'

sample_user_nums = 100000


# all_click_df指的是训练集
# sample_user_nums 采样作为验证集的用户数量
def trn_val_split(all_click_df, sample_user_nums):
    all_click = all_click_df
    all_user_ids = all_click.user_id.unique()

    # replace=True表示可以重复抽样，反之不可以
    sample_user_ids = np.random.choice(all_user_ids, size=sample_user_nums, replace=False)

    click_val = all_click[all_click['user_id'].isin(sample_user_ids)]
    click_trn = all_click[~all_click['user_id'].isin(sample_user_ids)]

    # 将验证集中的最后一次点击给抽取出来作为答案
    click_val = click_val.sort_values(['user_id', 'click_timestamp'])
    val_ans = click_val.groupby('user_id').tail(1)

    click_val = click_val.groupby('user_id').apply(lambda x: x[:-1]).reset_index(drop=True)

    # 去除val_ans中某些用户只有一个点击数据的情况，如果该用户只有一个点击数据，又被分到ans中，
    # 那么训练集中就没有这个用户的点击数据，出现用户冷启动问题，给自己模型验证带来麻烦
    val_ans = val_ans[val_ans.user_id.isin(click_val.user_id.unique())]  # 保证答案中出现的用户再验证集中还有
    click_val = click_val[click_val.user_id.isin(val_ans.user_id.unique())]

    return click_trn, click_val, val_ans


# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df


def get_trn_val_tst_data(data_path, offline=True):
    if offline:
        click_trn_data = pd.read_csv(data_path + 'train_click_log.csv')  # 训练集用户点击日志
        click_trn_data = reduce_mem(click_trn_data)
        click_trn, click_val, val_ans = trn_val_split(all_click_df, sample_user_nums)
    else:
        click_trn = pd.read_csv(data_path + 'train_click_log.csv')
        click_trn = reduce_mem(click_trn)
        click_val = None
        val_ans = None

    click_tst = pd.read_csv(data_path + 'testA_click_log.csv')

    return click_trn, click_val, click_tst, val_ans


# 返回多路召回列表或者单路召回
def get_recall_list(save_path, single_recall_model=None, multi_recall=False):
    if multi_recall:
        return pickle.load(open(save_path + 'final_recall_items_dict.pkl', 'rb'))

    if single_recall_model == 'i2i_itemcf':
        return pickle.load(open(save_path + 'itemcf_recall_dict.pkl', 'rb'))
    elif single_recall_model == 'i2i_emb_itemcf':
        return pickle.load(open(save_path + 'itemcf_emb_dict.pkl', 'rb'))
    elif single_recall_model == 'user_cf':
        return pickle.load(open(save_path + 'youtubednn_usercf_dict.pkl', 'rb'))
    elif single_recall_model == 'youtubednn':
        return pickle.load(open(save_path + 'youtube_u2i_dict.pkl', 'rb'))


def trian_item_word2vec(click_df, embed_size=64, save_name='item_w2v_emb.pkl', split_char=' '):
    click_df = click_df.sort_values('click_timestamp')
    # 只有转换成字符串才可以进行训练
    click_df['click_article_id'] = click_df['click_article_id'].astype(str)
    # 转换成句子的形式
    docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['click_article_id'].values.tolist()

    # 为了方便查看训练的进度，这里设定一个log信息
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

    # 这里的参数对训练得到的向量影响也很大,默认负采样为5
    w2v = Word2Vec(docs, size=16, sg=1, window=5, seed=2020, workers=24, min_count=1, iter=1)

    # 保存成字典的形式
    item_w2v_emb_dict = {k: w2v[k] for k in click_df['click_article_id']}
    pickle.dump(item_w2v_emb_dict, open(save_path + 'item_w2v_emb.pkl', 'wb'))

    return item_w2v_emb_dict


# 可以通过字典查询对应的item的Embedding
def get_embedding(save_path, all_click_df):
    if os.path.exists(save_path + 'item_content_emb.pkl'):
        item_content_emb_dict = pickle.load(open(save_path + 'item_content_emb.pkl', 'rb'))
    else:
        print('item_content_emb.pkl 文件不存在...')

    # w2v Embedding是需要提前训练好的
    if os.path.exists(save_path + 'item_w2v_emb.pkl'):
        item_w2v_emb_dict = pickle.load(open(save_path + 'item_w2v_emb.pkl', 'rb'))
    else:
        item_w2v_emb_dict = trian_item_word2vec(all_click_df)

    if os.path.exists(save_path + 'item_youtube_emb.pkl'):
        item_youtube_emb_dict = pickle.load(open(save_path + 'item_youtube_emb.pkl', 'rb'))
    else:
        print('item_youtube_emb.pkl 文件不存在...')

    if os.path.exists(save_path + 'user_youtube_emb.pkl'):
        user_youtube_emb_dict = pickle.load(open(save_path + 'user_youtube_emb.pkl', 'rb'))
    else:
        print('user_youtube_emb.pkl 文件不存在...')

    return item_content_emb_dict, item_w2v_emb_dict, item_youtube_emb_dict, user_youtube_emb_dict


def get_article_info_df():
    article_info_df = pd.read_csv(data_path + 'articles.csv')
    article_info_df = reduce_mem(article_info_df)

    return article_info_df


# 这里offline的online的区别就是验证集是否为空
click_trn, click_val, click_tst, val_ans = get_trn_val_tst_data(data_path, offline=False)

click_trn_hist, click_trn_last = get_hist_and_last_click(click_trn)

if click_val is not None:
    click_val_hist, click_val_last = click_val, val_ans
else:
    click_val_hist, click_val_last = None, None

click_tst_hist = click_tst


# 将召回列表转换成df的形式
def recall_dict_2_df(recall_list_dict):
    df_row_list = []  # [user, item, score]
    for user, recall_list in tqdm(recall_list_dict.items()):
        for item, score in recall_list:
            df_row_list.append([user, item, score])

    col_names = ['user_id', 'sim_item', 'score']
    recall_list_df = pd.DataFrame(df_row_list, columns=col_names)

    return recall_list_df


# 负采样函数，这里可以控制负采样时的比例, 这里给了一个默认的值
def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
    pos_data = recall_items_df[recall_items_df['label'] == 1]
    neg_data = recall_items_df[recall_items_df['label'] == 0]

    print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data), 'pos/neg:', len(pos_data) / len(neg_data))

    # 分组采样函数
    def neg_sample_func(group_df):
        neg_num = len(group_df)
        sample_num = max(int(neg_num * sample_rate), 1)  # 保证最少有一个
        sample_num = min(sample_num, 5)  # 保证最多不超过5个，这里可以根据实际情况进行选择
        return group_df.sample(n=sample_num, replace=True)

    # 对用户进行负采样，保证所有用户都在采样后的数据中
    neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
    # 对文章进行负采样，保证所有文章都在采样后的数据中
    neg_data_item_sample = neg_data.groupby('sim_item', group_keys=False).apply(neg_sample_func)

    # 将上述两种情况下的采样数据合并
    neg_data_new = neg_data_user_sample.append(neg_data_item_sample)
    # 由于上述两个操作是分开的，可能将两个相同的数据给重复选择了，所以需要对合并后的数据进行去重
    neg_data_new = neg_data_new.sort_values(['user_id', 'score']).drop_duplicates(['user_id', 'sim_item'], keep='last')

    # 将正样本数据合并
    data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)

    return data_new


# 召回数据打标签
def get_rank_label_df(recall_list_df, label_df, is_test=False):
    # 测试集是没有标签了，为了后面代码同一一些，这里直接给一个负数替代
    if is_test:
        recall_list_df['label'] = -1
        return recall_list_df

    label_df = label_df.rename(columns={'click_article_id': 'sim_item'})
    recall_list_df_ = recall_list_df.merge(label_df[['user_id', 'sim_item', 'click_timestamp']], \
                                           how='left', on=['user_id', 'sim_item'])
    recall_list_df_['label'] = recall_list_df_['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
    del recall_list_df_['click_timestamp']

    return recall_list_df_


def get_user_recall_item_label_df(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last, click_val_last,
                                  recall_list_df):
    # 获取训练数据的召回列表
    trn_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_trn_hist['user_id'].unique())]
    # 训练数据打标签
    trn_user_item_label_df = get_rank_label_df(trn_user_items_df, click_trn_last, is_test=False)
    # 训练数据负采样
    trn_user_item_label_df = neg_sample_recall_data(trn_user_item_label_df)

    if click_val is not None:
        val_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_val_hist['user_id'].unique())]
        val_user_item_label_df = get_rank_label_df(val_user_items_df, click_val_last, is_test=False)
        val_user_item_label_df = neg_sample_recall_data(val_user_item_label_df)
    else:
        val_user_item_label_df = None

    # 测试数据不需要进行负采样，直接对所有的召回商品进行打-1标签
    tst_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_tst_hist['user_id'].unique())]
    tst_user_item_label_df = get_rank_label_df(tst_user_items_df, None, is_test=True)

    return trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df


# 读取召回列表
recall_list_dict = get_recall_list(save_path, single_recall_model='i2i_itemcf')  # 这里只选择了单路召回的结果，也可以选择多路召回结果
# 将召回数据转换成df
recall_list_df = recall_dict_2_df(recall_list_dict)

# 给训练验证数据打标签，并负采样（这一部分时间比较久）
trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df = get_user_recall_item_label_df(click_trn_hist,
                                                                                                       click_val_hist,
                                                                                                       click_tst_hist,
                                                                                                       click_trn_last,
                                                                                                       click_val_last,
                                                                                                       recall_list_df)


# 将最终的召回的df数据转换成字典的形式做排序特征
def make_tuple_func(group_df):
    row_data = []
    for name, row_df in group_df.iterrows():
        row_data.append((row_df['sim_item'], row_df['score'], row_df['label']))

    return row_data


trn_user_item_label_tuples = trn_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
trn_user_item_label_tuples_dict = dict(zip(trn_user_item_label_tuples['user_id'], trn_user_item_label_tuples[0]))

if val_user_item_label_df is not None:
    val_user_item_label_tuples = val_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
    val_user_item_label_tuples_dict = dict(zip(val_user_item_label_tuples['user_id'], val_user_item_label_tuples[0]))
else:
    val_user_item_label_tuples_dict = None

tst_user_item_label_tuples = tst_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
tst_user_item_label_tuples_dict = dict(zip(tst_user_item_label_tuples['user_id'], tst_user_item_label_tuples[0]))


# 下面基于data做历史相关的特征
def create_feature(users_id, recall_list, click_hist_df, articles_info, articles_emb, user_emb=None, N=1):
    """
    基于用户的历史行为做相关特征
    :param users_id: 用户id
    :param recall_list: 对于每个用户召回的候选文章列表
    :param click_hist_df: 用户的历史点击信息
    :param articles_info: 文章信息
    :param articles_emb: 文章的embedding向量, 这个可以用item_content_emb, item_w2v_emb, item_youtube_emb
    :param user_emb: 用户的embedding向量， 这个是user_youtube_emb, 如果没有也可以不用， 但要注意如果要用的话， articles_emb就要用item_youtube_emb的形式， 这样维度才一样
    :param N: 最近的N次点击  由于testA日志里面很多用户只存在一次历史点击， 所以为了不产生空值，默认是1
    """

    # 建立一个二维列表保存结果， 后面要转成DataFrame
    all_user_feas = []
    i = 0
    for user_id in tqdm(users_id):
        # 该用户的最后N次点击
        hist_user_items = click_hist_df[click_hist_df['user_id'] == user_id]['click_article_id'][-N:]

        # 遍历该用户的召回列表
        for rank, (article_id, score, label) in enumerate(recall_list[user_id]):
            # 该文章建立时间, 字数
            a_create_time = articles_info[articles_info['article_id'] == article_id]['created_at_ts'].values[0]
            a_words_count = articles_info[articles_info['article_id'] == article_id]['words_count'].values[0]
            single_user_fea = [user_id, article_id]
            # 计算与最后点击的商品的相似度的和， 最大值和最小值， 均值
            sim_fea = []
            time_fea = []
            word_fea = []
            # 遍历用户的最后N次点击文章
            for hist_item in hist_user_items:
                b_create_time = articles_info[articles_info['article_id'] == hist_item]['created_at_ts'].values[0]
                b_words_count = articles_info[articles_info['article_id'] == hist_item]['words_count'].values[0]

                sim_fea.append(np.dot(articles_emb[hist_item], articles_emb[article_id]))
                time_fea.append(abs(a_create_time - b_create_time))
                word_fea.append(abs(a_words_count - b_words_count))

            single_user_fea.extend(sim_fea)  # 相似性特征
            single_user_fea.extend(time_fea)  # 时间差特征
            single_user_fea.extend(word_fea)  # 字数差特征
            single_user_fea.extend([max(sim_fea), min(sim_fea), sum(sim_fea), sum(sim_fea) / len(sim_fea)])  # 相似性的统计特征

            if user_emb:  # 如果用户向量有的话， 这里计算该召回文章与用户的相似性特征
                single_user_fea.append(np.dot(user_emb[user_id], articles_emb[article_id]))

            single_user_fea.extend([score, rank, label])
            # 加入到总的表中
            all_user_feas.append(single_user_fea)

    # 定义列名
    id_cols = ['user_id', 'click_article_id']
    sim_cols = ['sim' + str(i) for i in range(N)]
    time_cols = ['time_diff' + str(i) for i in range(N)]
    word_cols = ['word_diff' + str(i) for i in range(N)]
    sat_cols = ['sim_max', 'sim_min', 'sim_sum', 'sim_mean']
    user_item_sim_cols = ['user_item_sim'] if user_emb else []
    user_score_rank_label = ['score', 'rank', 'label']
    cols = id_cols + sim_cols + time_cols + word_cols + sat_cols + user_item_sim_cols + user_score_rank_label

    # 转成DataFrame
    df = pd.DataFrame(all_user_feas, columns=cols)

    return df


article_info_df = get_article_info_df()
all_click = click_trn.append(click_tst)
item_content_emb_dict, item_w2v_emb_dict, item_youtube_emb_dict, user_youtube_emb_dict = get_embedding(save_path,
                                                                                                       all_click)

# 获取训练验证及测试数据中召回列文章相关特征
trn_user_item_feats_df = create_feature(trn_user_item_label_tuples_dict.keys(), trn_user_item_label_tuples_dict, \
                                        click_trn_hist, article_info_df, item_content_emb_dict)

if val_user_item_label_tuples_dict is not None:
    val_user_item_feats_df = create_feature(val_user_item_label_tuples_dict.keys(), val_user_item_label_tuples_dict, \
                                            click_val_hist, article_info_df, item_content_emb_dict)
else:
    val_user_item_feats_df = None

tst_user_item_feats_df = create_feature(tst_user_item_label_tuples_dict.keys(), tst_user_item_label_tuples_dict, \
                                        click_tst_hist, article_info_df, item_content_emb_dict)

# 保存一份省的每次都要重新跑，每次跑的时间都比较长
trn_user_item_feats_df.to_csv(save_path + 'trn_user_item_feats_df.csv', index=False)

if val_user_item_feats_df is not None:
    val_user_item_feats_df.to_csv(save_path + 'val_user_item_feats_df.csv', index=False)

tst_user_item_feats_df.to_csv(save_path + 'tst_user_item_feats_df.csv', index=False)

click_tst.head()

# 读取文章特征
articles = pd.read_csv(data_path + 'articles.csv')
articles = reduce_mem(articles)

# 日志数据，就是前面的所有数据
if click_val is not None:
    all_data = click_trn.append(click_val)
all_data = click_trn.append(click_tst)
all_data = reduce_mem(all_data)

# 拼上文章信息
all_data = all_data.merge(articles, left_on='click_article_id', right_on='article_id')

all_data.shape


def active_level(all_data, cols):
    """
    制作区分用户活跃度的特征
    :param all_data: 数据集
    :param cols: 用到的特征列
    """
    data = all_data[cols]
    data.sort_values(['user_id', 'click_timestamp'], inplace=True)
    user_act = pd.DataFrame(data.groupby('user_id', as_index=False)[['click_article_id', 'click_timestamp']]. \
                            agg({'click_article_id': np.size, 'click_timestamp': {list}}).values,
                            columns=['user_id', 'click_size', 'click_timestamp'])

    # 计算时间间隔的均值
    def time_diff_mean(l):
        if len(l) == 1:
            return 1
        else:
            return np.mean([j - i for i, j in list(zip(l[:-1], l[1:]))])

    user_act['time_diff_mean'] = user_act['click_timestamp'].apply(lambda x: time_diff_mean(x))

    # 点击次数取倒数
    user_act['click_size'] = 1 / user_act['click_size']

    # 两者归一化
    user_act['click_size'] = (user_act['click_size'] - user_act['click_size'].min()) / (
            user_act['click_size'].max() - user_act['click_size'].min())
    user_act['time_diff_mean'] = (user_act['time_diff_mean'] - user_act['time_diff_mean'].min()) / (
            user_act['time_diff_mean'].max() - user_act['time_diff_mean'].min())
    user_act['active_level'] = user_act['click_size'] + user_act['time_diff_mean']

    user_act['user_id'] = user_act['user_id'].astype('int')
    del user_act['click_timestamp']

    return user_act


user_act_fea = active_level(all_data, ['user_id', 'click_article_id', 'click_timestamp'])

user_act_fea.head()


def hot_level(all_data, cols):
    """
    制作衡量文章热度的特征
    :param all_data: 数据集
    :param cols: 用到的特征列
    """
    data = all_data[cols]
    data.sort_values(['click_article_id', 'click_timestamp'], inplace=True)
    article_hot = pd.DataFrame(data.groupby('click_article_id', as_index=False)[['user_id', 'click_timestamp']]. \
                               agg({'user_id': np.size, 'click_timestamp': {list}}).values,
                               columns=['click_article_id', 'user_num', 'click_timestamp'])

    # 计算被点击时间间隔的均值
    def time_diff_mean(l):
        if len(l) == 1:
            return 1
        else:
            return np.mean([j - i for i, j in list(zip(l[:-1], l[1:]))])

    article_hot['time_diff_mean'] = article_hot['click_timestamp'].apply(lambda x: time_diff_mean(x))

    # 点击次数取倒数
    article_hot['user_num'] = 1 / article_hot['user_num']

    # 两者归一化
    article_hot['user_num'] = (article_hot['user_num'] - article_hot['user_num'].min()) / (
            article_hot['user_num'].max() - article_hot['user_num'].min())
    article_hot['time_diff_mean'] = (article_hot['time_diff_mean'] - article_hot['time_diff_mean'].min()) / (
            article_hot['time_diff_mean'].max() - article_hot['time_diff_mean'].min())
    article_hot['hot_level'] = article_hot['user_num'] + article_hot['time_diff_mean']

    article_hot['click_article_id'] = article_hot['click_article_id'].astype('int')

    del article_hot['click_timestamp']

    return article_hot


article_hot_fea = hot_level(all_data, ['user_id', 'click_article_id', 'click_timestamp'])

article_hot_fea.head()


def device_fea(all_data, cols):
    """
    制作用户的设备特征
    :param all_data: 数据集
    :param cols: 用到的特征列
    """
    user_device_info = all_data[cols]

    # 用众数来表示每个用户的设备信息
    user_device_info = user_device_info.groupby('user_id').agg(lambda x: x.value_counts().index[0]).reset_index()

    return user_device_info


# 设备特征(这里时间会比较长)
device_cols = ['user_id', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region',
               'click_referrer_type']
user_device_info = device_fea(all_data, device_cols)

user_device_info.head()


def user_time_hob_fea(all_data, cols):
    """
    制作用户的时间习惯特征
    :param all_data: 数据集
    :param cols: 用到的特征列
    """
    user_time_hob_info = all_data[cols]

    # 先把时间戳进行归一化
    mm = MinMaxScaler()
    user_time_hob_info['click_timestamp'] = mm.fit_transform(user_time_hob_info[['click_timestamp']])
    user_time_hob_info['created_at_ts'] = mm.fit_transform(user_time_hob_info[['created_at_ts']])

    user_time_hob_info = user_time_hob_info.groupby('user_id').agg('mean').reset_index()

    user_time_hob_info.rename(columns={'click_timestamp': 'user_time_hob1', 'created_at_ts': 'user_time_hob2'},
                              inplace=True)
    return user_time_hob_info


user_time_hob_cols = ['user_id', 'click_timestamp', 'created_at_ts']
user_time_hob_info = user_time_hob_fea(all_data, user_time_hob_cols)


def user_cat_hob_fea(all_data, cols):
    """
    用户的主题爱好
    :param all_data: 数据集
    :param cols: 用到的特征列
    """
    user_category_hob_info = all_data[cols]
    user_category_hob_info = user_category_hob_info.groupby('user_id').agg({list}).reset_index()

    user_cat_hob_info = pd.DataFrame()
    user_cat_hob_info['user_id'] = user_category_hob_info['user_id']
    user_cat_hob_info['cate_list'] = user_category_hob_info['category_id']

    return user_cat_hob_info


user_category_hob_cols = ['user_id', 'category_id']
user_cat_hob_info = user_cat_hob_fea(all_data, user_category_hob_cols)

user_wcou_info = all_data.groupby('user_id')['words_count'].agg('mean').reset_index()
user_wcou_info.rename(columns={'words_count': 'words_hbo'}, inplace=True)

# 所有表进行合并
user_info = pd.merge(user_act_fea, user_device_info, on='user_id')
user_info = user_info.merge(user_time_hob_info, on='user_id')
user_info = user_info.merge(user_cat_hob_info, on='user_id')
user_info = user_info.merge(user_wcou_info, on='user_id')

# 这样用户特征以后就可以直接读取了
user_info.to_csv(save_path + 'user_info.csv', index=False)

# 把用户信息直接读入进来
user_info = pd.read_csv(save_path + 'user_info.csv')

if os.path.exists(save_path + 'trn_user_item_feats_df.csv'):
    trn_user_item_feats_df = pd.read_csv(save_path + 'trn_user_item_feats_df.csv')

if os.path.exists(save_path + 'tst_user_item_feats_df.csv'):
    tst_user_item_feats_df = pd.read_csv(save_path + 'tst_user_item_feats_df.csv')

if os.path.exists(save_path + 'val_user_item_feats_df.csv'):
    val_user_item_feats_df = pd.read_csv(save_path + 'val_user_item_feats_df.csv')
else:
    val_user_item_feats_df = None

# 拼上用户特征
# 下面是线下验证的
trn_user_item_feats_df = trn_user_item_feats_df.merge(user_info, on='user_id', how='left')

if val_user_item_feats_df is not None:
    val_user_item_feats_df = val_user_item_feats_df.merge(user_info, on='user_id', how='left')
else:
    val_user_item_feats_df = None

tst_user_item_feats_df = tst_user_item_feats_df.merge(user_info, on='user_id', how='left')

trn_user_item_feats_df.columns

Index(['user_id', 'click_article_id', 'sim0', 'time_diff0', 'word_diff0',
       'sim_max', 'sim_min', 'sim_sum', 'sim_mean', 'score', 'rank', 'label',
       'click_size', 'time_diff_mean', 'active_level', 'click_environment',
       'click_deviceGroup', 'click_os', 'click_country', 'click_region',
       'click_referrer_type', 'user_time_hob1', 'user_time_hob2', 'cate_list',
       'words_hbo'],
      dtype='object')

articles = pd.read_csv(data_path + 'articles.csv')
articles = reduce_mem(articles)

# 拼上文章特征
trn_user_item_feats_df = trn_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')

if val_user_item_feats_df is not None:
    val_user_item_feats_df = val_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')
else:
    val_user_item_feats_df = None

tst_user_item_feats_df = tst_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')

trn_user_item_feats_df['is_cat_hab'] = trn_user_item_feats_df.apply(
    lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1)
if val_user_item_feats_df is not None:
    val_user_item_feats_df['is_cat_hab'] = val_user_item_feats_df.apply(
        lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1)
else:
    val_user_item_feats_df = None
tst_user_item_feats_df['is_cat_hab'] = tst_user_item_feats_df.apply(
    lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1)

# 线下验证
del trn_user_item_feats_df['cate_list']

if val_user_item_feats_df is not None:
    del val_user_item_feats_df['cate_list']
else:
    val_user_item_feats_df = None

del tst_user_item_feats_df['cate_list']

del trn_user_item_feats_df['article_id']

if val_user_item_feats_df is not None:
    del val_user_item_feats_df['article_id']
else:
    val_user_item_feats_df = None

del tst_user_item_feats_df['article_id']

# 训练验证特征
trn_user_item_feats_df.to_csv(save_path + 'trn_user_item_feats_df.csv', index=False)
if val_user_item_feats_df is not None:
    val_user_item_feats_df.to_csv(save_path + 'val_user_item_feats_df.csv', index=False)
tst_user_item_feats_df.to_csv(save_path + 'tst_user_item_feats_df.csv', index=False)
