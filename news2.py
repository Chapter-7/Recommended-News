import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
from sklearn.preprocessing import MinMaxScaler

# 设置显示中文字体
plt.rc('font', family='SimHei', size=13)
warnings.filterwarnings("ignore")

# 定义内存释放函数
def release_memory():
    plt.clf()
    gc.collect()

# 读取数据
path = '../data/'
trn_click = pd.read_csv(path + 'train_click_log.csv')
item_df = pd.read_csv(path + 'articles.csv')
item_df = item_df.rename(columns={'article_id': 'click_article_id'})  # 重命名，方便后续match
item_emb_df = pd.read_csv(path + 'articles_emb.csv')
tst_click = pd.read_csv(path + 'testA_click_log.csv')

# 数据预处理
trn_click['rank'] = trn_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
tst_click['rank'] = tst_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
trn_click['click_cnts'] = trn_click.groupby(['user_id'])['click_timestamp'].transform('count')
tst_click['click_cnts'] = tst_click.groupby(['user_id'])['click_timestamp'].transform('count')

# 合并 item_df 数据
trn_click = trn_click.merge(item_df, how='left', on=['click_article_id'])
tst_click = tst_click.merge(item_df, how='left', on=['click_article_id'])

# 用户点击日志信息
print(trn_click.info())
print(trn_click.describe())

# 限制绘制的数据量为前 20 条
plt.figure(figsize=(15, 20), dpi=80)
i = 1
for col in ['click_article_id', 'click_timestamp', 'click_environment', 'click_deviceGroup', 'click_os',
            'click_country', 'click_region', 'click_referrer_type', 'rank', 'click_cnts']:
    plt.subplot(5, 2, i)
    i += 1
    v = trn_click[col].value_counts().reset_index().head(20)
    v.columns = ['index', 'count']
    fig = sns.barplot(x=v['index'], y=v['count'])
    for item in fig.get_xticklabels():
        item.set_rotation(90)
    plt.title(col)

plt.tight_layout()
plt.show()

release_memory()

# 测试集用户点击日志
print(tst_click.describe())

# 用户点击日志合并
user_click_merge = pd.concat([trn_click, tst_click], ignore_index=True)

# 新闻点击次数分析
item_click_count = sorted(user_click_merge.groupby('click_article_id')['user_id'].count(), reverse=True)
plt.plot(item_click_count[:100])
plt.show()

release_memory()

# 新闻共现频次：两篇新闻连续出现的次数
tmp = user_click_merge.sort_values('click_timestamp')
tmp['next_item'] = tmp.groupby(['user_id'])['click_article_id'].shift(-1)
union_item = tmp.groupby(['click_article_id', 'next_item'])['click_timestamp'].size().reset_index(name='count').sort_values('count', ascending=False)
print(union_item[['count']].describe())
plt.scatter(union_item['click_article_id'], union_item['count'])
plt.show()

release_memory()

# 点击时间差的平均值
mm = MinMaxScaler()
user_click_merge[['click_timestamp', 'created_at_ts']] = mm.fit_transform(user_click_merge[['click_timestamp', 'created_at_ts']])
user_click_merge = user_click_merge.sort_values('click_timestamp')

def mean_diff_time_func(df, col):
    df = pd.DataFrame(df, columns=[col])
    df['time_shift1'] = df[col].shift(1).fillna(0)
    df['diff_time'] = abs(df[col] - df['time_shift1'])
    return df['diff_time'].mean()

mean_diff_click_time = user_click_merge.groupby('user_id').apply(lambda x: mean_diff_time_func(x, 'click_timestamp'))
plt.plot(sorted(mean_diff_click_time.values, reverse=True))
plt.show()

release_memory()

# 新闻embedding向量表示
item_emb_df = item_emb_df.rename(columns={'article_id': 'click_article_id'})
item_idx_2_rawid_dict = dict(zip(item_emb_df['click_article_id'], item_emb_df.index))
del item_emb_df['click_article_id']
item_emb_np = np.ascontiguousarray(item_emb_df.values, dtype=np.float32)

# 用户点击新闻相似度分析
sub_user_ids = np.random.choice(user_click_merge.user_id.unique(), size=15, replace=False)
sub_user_info = user_click_merge[user_click_merge['user_id'].isin(sub_user_ids)]

def get_item_sim_list(df):
    sim_list = []
    item_list = df['click_article_id'].values
    for i in range(0, len(item_list)-1):
        emb1 = item_emb_np[item_idx_2_rawid_dict[item_list[i]]]
        emb2 = item_emb_np[item_idx_2_rawid_dict[item_list[i+1]]]
        sim_list.append(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    return sim_list

for _, user_df in sub_user_info.groupby('user_id'):
    item_sim_list = get_item_sim_list(user_df)
    plt.plot(item_sim_list)
plt.show()

release_memory()

# 随机用户的点击环境分布
def plot_envs(df, cols, r, c):
    plt.figure(figsize=(10, 5), dpi=80)
    i = 1
    for col in cols:
        plt.subplot(r, c, i)
        i += 1
        v = df[col].value_counts().reset_index().head(20)
        v.columns = ['category', 'count']
        fig = sns.barplot(x=v['category'], y=v['count'])
        for item in fig.get_xticklabels():
            item.set_rotation(90)
        plt.title(col)
    plt.tight_layout()
    plt.show()

sample_user_ids = np.random.choice(tst_click['user_id'].unique(), size=5, replace=False)
sample_users = user_click_merge[user_click_merge['user_id'].isin(sample_user_ids)]
cols = ['click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type']
for _, user_df in sample_users.groupby('user_id'):
    plot_envs(user_df, cols, 2, 3)
    release_memory()

# 用户点击新闻数量的分布
user_click_item_count = sorted(user_click_merge.groupby('user_id')['click_article_id'].count(), reverse=True)
plt.plot(user_click_item_count[:50])
plt.show()

release_memory()
