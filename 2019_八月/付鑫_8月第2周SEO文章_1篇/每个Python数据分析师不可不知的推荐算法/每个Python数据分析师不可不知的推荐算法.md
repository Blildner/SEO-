每个Python数据分析师不可不知的推荐算法
======================================

[TOC]





BASELINE
========

基于Baseline的推荐算法 

​        要评估一个策略的好坏，就需要建立一个对比基线，以便后续观察算法效果的提升。此处我们可以简单地对推荐算法进行建模作为基线。假设我们的训练数据为：  <user, item, rating>三元组， 其中user为用户id， item为物品id(item可以是MovieLens上的电影，Amazon上的书， 或是百度关键词工具上的关键词), rating为user对item的投票分数， 其中用户u对物品i的真实投票分数我们记为rui，基线(baseline)模型预估分数为bui，则可建模如下：


$$
b_{u i}=\mu+b_{u}+b_{i}
$$
​	其中mu（希腊字母mu）为所有已知投票数据中投票的均值，bu为用户的打分相对于平均值的偏差（如果某用户比较苛刻，打分都相对偏低， 则bu会为负值；相反，如果某用户经常对很多片都打正分， 则bu为正值）； bi为该item被打分时，相对于平均值得偏差，可反映电影受欢迎程度。 bui则为基线模型对用户u给物品i打分的预估值。该模型虽然简单， 但其中其实已经包含了用户个性化和item的个性化信息， 而且特别简单（很多时候， 简单就是一个非常大的特点， 特别是面对大规模数据时）。

​         基线模型中， mu可以直接统计得到，我们的优化函数可以写为（其实就是最小二乘法）：
$$
\min _{b_{*}} \sum_{(u, i) \in \mathcal{K}}\left(r_{u i}-\mu-b_{u}-b_{i}\right)^{2}+\lambda_{1}\left(\sum_{u} b_{u}^{2}+\sum_{i} b_{i}^{2}\right)
$$


​        也可以直接写成如下式子，因为它本身就是经验似然：
$$
\begin{aligned} b_{i} &=\frac{\sum_{u \in \mathrm{R}(i)}\left(r_{u i}-\mu\right)}{\lambda_{2}+|\mathrm{R}(i)|} \\ b_{u} &=\frac{\sum_{i \in \mathrm{R}(u)}\left(r_{u i}-\mu-b_{i}\right)}{\lambda_{3}+|\mathrm{R}(u)|} \end{aligned}
$$


```python
test = pd.read_csv('../data/Antai_AE_round1_test_20190626.csv')
tmp = test[test['irank']<=31].sort_values(by=['buyer_country_id', 'buyer_admin_id', 'irank'])[['buyer_admin_id','item_id','irank']]
sub = tmp.set_index(['buyer_admin_id', 'irank']).unstack(-1)
sub.fillna(5595070).astype(int).reset_index().to_csv('../submit/sub.csv', index=False, header=None)
```

```python
# 最终提交文件格式
sub = pd.read_csv('../submit/sub.csv', header = None)
sub.head()
```

协同过滤
========



userCF
------


$$
\operatorname{sim}(i, j)=\frac{\sum_{x \in I_{ij}}\left(R_{i, x}-\overline{R}_{i}\right)\left(R_{j, x}-\overline{R_{j}}\right)}{\sqrt{\sum_{x \in I_{ij}}\left(R_{i, x}-\overline{R}_{i}\right)^{2}} \sqrt{\sum_{x \in I_{ij}}\left(R_{j, x}-\overline{R}_{j}\right)^{2}}}
$$

该公式要计算用户i和用户j之间的相似度, I(ij)是代表用户i和用户j共同评价过的物品, R(i,x)代表用户i对物品x的评分, R(i)头上有一杠的代表用户i所有评分的平均分, 之所以要减去平均分是因为有的用户打分严有的松, 归一化用户打分避免相互影响。

```python

import datetime
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示
from zipfile import ZipFile
```

```python
import pickle
usingSaved=True
```

```python
#读取item
if not usingSaved:
    output = open('item_attr.pkl', 'wb')
    myzip=ZipFile('data/Antai_AE_round1_item_attr_20190626.zip')
    f=myzip.open('Antai_AE_round1_item_attr_20190626.csv')
    item_attr=pd.read_csv(f)
    f.close()
    myzip.close()
    pickle.dump(item_attr, output)
    output.close()
else:
    item_attr = pickle.load(open('item_attr.pkl', 'rb'), encoding='iso-8859-1')
#     item_attr = pickle.load(open('item_attr.pkl', 'rb'))

#读取train  
if not usingSaved:
    output = open('train.pkl', 'wb')
    myzip=ZipFile('data/Antai_AE_round1_train_20190626.zip')
    f=myzip.open('Antai_AE_round1_train_20190626.csv')
    train=pd.read_csv(f)
    f.close()
    myzip.close()
    train['create_order_time'] = train.create_order_time.apply(lambda x:pd.to_datetime(x))
    train['hour']=train['create_order_time'].dt.hour
    train['date']=train['create_order_time'].dt.day
    train['month']=train['create_order_time'].dt.month
    train['year']=train['create_order_time'].dt.year
    train['month-date'] = train.month.astype(str)+'-'+train.date.astype(str)
    train['count'] = 1
    train['dayofweek']=train['create_order_time'].dt.dayofweek
    train['isweekend']=train['dayofweek'].apply(lambda x:0 if x<5 else 1)
    pickle.dump(train, output)
    output.close()
else:
    train = pickle.load(open('train.pkl', 'rb'),encoding='iso-8859-1')
#     train = pickle.load(open('train.pkl', 'rb'))
    
#读取test
if not usingSaved:
    output = open('test.pkl', 'wb')
    test=pd.read_csv('data/Antai_AE_round1_test_20190626.csv')
    test['create_order_time'] = test.create_order_time.apply(lambda x:pd.to_datetime(x))
    test['hour']=test['create_order_time'].dt.hour
    test['date']=test['create_order_time'].dt.day
    test['month']=test['create_order_time'].dt.month
    test['year']=test['create_order_time'].dt.year
    test['count'] = 1
    test['month-date'] = test.month.astype(str)+'-'+test.date.astype(str)
    test['dayofweek']=test['create_order_time'].dt.dayofweek
    test['isweekend']=test['dayofweek'].apply(lambda x:0 if x<5 else 1)
    pickle.dump(test, output)
    output.close()
else:
    test = pickle.load(open('test.pkl', 'rb'), encoding='iso-8859-1')
submit = pd.read_csv('data/Antai_AE_round1_submit_20190715.csv')
#     test = pickle.load(open('test.pkl', 'rb'))
```

- yy国 有138678用户，926771item

```
trainyy = train[train.buyer_country_id =='yy']
submit.head()
```

```
merge_yy = pd.concat([trainyy,test])

merge_yy.item_id.unique().shape,merge_yy.buyer_admin_id.unique().shape
```

- 融合 yy过的train和test一起学习usercf
- 推荐时候可能会重复购买

```python
import sys
import random
import math
import os
from operator import itemgetter

from collections import defaultdict

random.seed(0)

class UserBasedCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering '''

    # 构造函数，用来初始化
    def __init__(self):
        # 定义 训练集 测试集 为字典类型
        self.trainset = {}
        self.testset = {}
        # 训练集用的相似用户数
        self.n_sim_user = 30
        # 推荐Item数量
        self.n_rec_item = 30

        self.user_sim_mat = {}
        self.item_popular = {}
        self.item_count = 0
        # sys.stderr 是用来重定向标准错误信息的
        print ('相似用户数目为 = %d' % self.n_sim_user, file=sys.stderr)
        print ('推荐Item数目为 = %d' %
               self.n_rec_item, file=sys.stderr)

    # 划分训练集和测试集 pivot用来定义训练集和测试集的比例
    def generate_dataset(self, train,test=None, pivot=0.90):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0
        if test is None:# 随机分配验证机实验 
            print('算法尝试！')
            for line in train.iterrows():
                user, item, rating = line[1][0],line[1][1],1
                # split the data by pivot
                if random.random() < pivot:
                    self.trainset.setdefault(user, {})
                    self.trainset[user][item] = int(rating)
                    trainset_len += 1
                else:
                    self.testset.setdefault(user, {})
                    self.testset[user][item] = int(rating)
                    testset_len += 1
        else:#真正预测
            print('预测尝试！')
            for line in train.iterrows():
                user, item, rating = line[1][0],line[1][1],1
                # split the data by pivot
                self.trainset.setdefault(user, {})
                self.trainset[user][item] = int(rating)
                trainset_len += 1
            del user,item,rating
            for line in test.iterrows():
                user, item, rating = line[1][0],line[1][1],1
                self.testset.setdefault(user, {})
                self.testset[user][item] = int(rating)
                testset_len += 1
            
        print ('split training set and test set succ')
        print ('train set = %s' % trainset_len)
        print ('test set = %s' % testset_len)
    # 建立物品-用户 倒排表
    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # build inverse table for item-users
        # key=itemID, value=list of userIDs who have seen this item
        print ('构建物品-用户倒排表中，请等待......', file=sys.stderr)
        item2users = dict()

        # Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组
        for user, items in self.trainset.items():
            for item in items:
                # inverse table for item-users
                if item not in item2users:
                    # 根据商品id 构造set() 函数创建一个无序不重复元素集
                    item2users[item] = set()
                # 集合中值为用户id
                # 数值形如
                # {'914': {'1','6','10'}, '3408': {'1'} ......}
                item2users[item].add(user)
                # 记录电影的流行度
                if item not in self.item_popular:
                    self.item_popular[item] = 0
                self.item_popular[item] += 1
        print ('构建物品-用户倒排表成功', file=sys.stderr)

        # save the total item number, which will be used in evaluation
        self.item_count = len(item2users)
        print ('总共被操作过的物品数目为 = %d' % self.item_count, file=sys.stderr)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat

        print ('building user co-rated items matrix...', file=sys.stderr)
        # 令系数矩阵 C[u][v]表示N(u)∩N（v) ，假如用户u和用户v同时属于K个物品对应的用户列表，就有C[u][v]=K
        for item, users in item2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1
        print ('build user co-rated items matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000
        # 循环遍历usersim_mat 根据余弦相似度公式计算出用户兴趣相似度
        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                # 以下是公式计算过程
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                #计数 并没有什么卵用
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print ('calculating user similarity factor(%d)' %
                           simfactor_count, file=sys.stderr)

        print ('calculate user similarity matrix(similarity factor) succ',
               file=sys.stderr)
        print ('Total similarity factor number = %d' %
               simfactor_count, file=sys.stderr)
    # 根据用户给予推荐结果
    def recommend(self, user,predict=False):
        '''定义给定K个相似用户和推荐N个商品'''
        K = self.n_sim_user
        N = self.n_rec_item
        # 定义一个字典来存储为用户推荐的电影
        rank = dict()
        
        watched_items = self.trainset[user]
        # sorted() 函数对所有可迭代的对象进行排序操作。 key 指定比较的对象 ，reverse=True 降序
        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for item in self.trainset[similar_user]:
                # 判断 如果这个商品 该用户已经买过 则跳出循环 ##此处不太成立
#                 if item in watched_items:
#                     continue
                # 记录用户对推荐的电影的兴趣度
                rank.setdefault(item, 0)
                rank[item] += similarity_factor
        # return the N best items
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]
    def predict(self):
        print('Predict start...')

        N = self.n_rec_item
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_items = set()
        # varables for popularity
        popular_sum = 0
        predict_result=[]
        for i, user in enumerate(self.testset):
            if i % 500 == 0:
                print ('recommended for %d users' % i)
            rec_items = self.recommend(user,predict=True)
            predict_result.append((user,rec_items))
        print('Predict end...')
        return predict_result
    # 计算 准确略，召回率，覆盖率，流行度
    def evaluate(self):

        ''' print evaluation result: precision, recall, coverage and popularity '''
        print ('Evaluation start...', file=sys.stderr)

        N = self.n_rec_item
        #  varables for precision and recall
        #记录推荐正确的电影数
        hit = 0
        #记录推荐电影的总数
        rec_count = 0
        #记录测试数据中总数
        test_count = 0
        # varables for coverage
        all_rec_items = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
            test_items = self.testset.get(user, {})
            rec_items = self.recommend(user,predict=False)
            for item, _ in rec_items:
                if item in test_items:
                    hit += 1
                all_rec_items.add(item)
                popular_sum += math.log(1 + self.item_popular[item])
            rec_count += N
            test_count += len(test_items)
        # 计算准确度
        precision = hit / (1.0 * rec_count)
        # 计算召回率
        recall = hit / (1.0 * test_count)
        # 计算覆盖率
        coverage = len(all_rec_items) / (1.0 * self.item_count)
        #计算流行度
        popularity = popular_sum / (1.0 * rec_count)

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
               (precision, recall, coverage, popularity), file=sys.stderr)
```

算法尝试

```python
usercf = UserBasedCF()
usercf.generate_dataset(merge_yy[['buyer_admin_id','item_id']])
usercf.calc_user_sim()
usercf.evaluate()
```

预测尝试

```python
usercf_pred = UserBasedCF()
usercf_pred.generate_dataset(merge_yy[['buyer_admin_id','item_id']],test[['buyer_admin_id','item_id']].drop_duplicates(subset=['buyer_admin_id'], keep='first'))
usercf_pred.calc_user_sim()
output = open('usercf_pred.pkl', 'wb')
pickle.dump(usercf_pred, output)
output.close()
result = usercf_pred.predict()
```

```python
result_df = pd.DataFrame(result,columns=['buyer_admin_id','rec_items'])
result_df['rec_items'] = result_df.rec_items.apply(lambda x:[i[0] for i in x])
result_df['len'] = result_df.rec_items.apply(lambda x:len(x))
result_df[result_df['len'] < 30].head()
```

使用活跃商品填补

```python
popularity_check = usercf_pred.item_popular
popularity_check_sorted = sorted(popularity_check.items(), key=lambda obj: obj[1],reverse=True) 
for i in range(30):
    result_df['predict '+str(i+1)] = result_df.rec_items.apply(lambda x:x[i] if len(x) >i else random.choice( popularity_check_sorted [:100])[0])
result_df[result_df['len']<20].head(2) # 已经填上了

result_df.buyer_admin_id.unique().shape # 跟测试集个数匹配

result_df.sort_values('buyer_admin_id').drop(['rec_items','len'],axis=1).to_csv('user_cf.csv',index=False,header=False)

```



itemCF
------


$$
\operatorname{sim}(i, j)=\frac{\sum_{x \in U_{ij}}\left(r_{i, x}-\overline{{r}_{i}}\right)\left(r_{j, x}-\overline{r_{j}}\right)}{\sqrt{\sum_{x \in U_{ij}}\left(r_{i, x}-\overline{{r}_{i}}\right)^{2}} \sqrt{\sum_{x \in U_{ij}}\left(r_{j, x}-\overline{{r}_{j}}\right)^{2}}}
$$
该公式要计算物品i和物品j之间的相似度, U(ij)是代表物品i和物品j同时评价过的用户集合, r(i,x)代表物品i被用户x的评分, r(i)头上有一杠的代表物品i被所有用户评分的平均分。



```python
import datetime
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示
from zipfile import ZipFile

import pickle
usingSaved=True
```

```python
#读取item
if not usingSaved:
    output = open('item_attr.pkl', 'wb')
    myzip=ZipFile('data/Antai_AE_round1_item_attr_20190626.zip')
    f=myzip.open('Antai_AE_round1_item_attr_20190626.csv')
    item_attr=pd.read_csv(f)
    f.close()
    myzip.close()
    pickle.dump(item_attr, output)
    output.close()
else:
    item_attr = pickle.load(open('item_attr.pkl', 'rb'), encoding='iso-8859-1')
#读取train  
if not usingSaved:
    output = open('train.pkl', 'wb')
    myzip=ZipFile('data/Antai_AE_round1_train_20190626.zip')
    f=myzip.open('Antai_AE_round1_train_20190626.csv')
    train=pd.read_csv(f)
    f.close()
    myzip.close()
    train['create_order_time'] = train.create_order_time.apply(lambda x:pd.to_datetime(x))
    train['hour']=train['create_order_time'].dt.hour
    train['date']=train['create_order_time'].dt.day
    train['month']=train['create_order_time'].dt.month
    train['year']=train['create_order_time'].dt.year
    train['month-date'] = train.month.astype(str)+'-'+train.date.astype(str)
    train['count'] = 1
    train['dayofweek']=train['create_order_time'].dt.dayofweek
    train['isweekend']=train['dayofweek'].apply(lambda x:0 if x<5 else 1)
    pickle.dump(train, output)
    output.close()
else:
    train = pickle.load(open('train.pkl', 'rb'),encoding='iso-8859-1')
#读取test
if not usingSaved:
    output = open('test.pkl', 'wb')
    test=pd.read_csv('data/Antai_AE_round1_test_20190626.csv')
    test['create_order_time'] = test.create_order_time.apply(lambda x:pd.to_datetime(x))
    test['hour']=test['create_order_time'].dt.hour
    test['date']=test['create_order_time'].dt.day
    test['month']=test['create_order_time'].dt.month
    test['year']=test['create_order_time'].dt.year
    test['month-date'] = test.month.astype(str)+'-'+test.date.astype(str)
    test['count'] = 1
    train['dayofweek']=train['create_order_time'].dt.dayofweek
    train['isweekend']=train['dayofweek'].apply(lambda x:0 if x<5 else 1)
    pickle.dump(test, output)
    output.close()
else:
    test = pickle.load(open('test.pkl', 'rb'), encoding='iso-8859-1')
```

```python
test.head()
item_attr.head()
train.head()
```

```python
import sys
import random
import math
import os
from operator import itemgetter
from collections import defaultdict

random.seed(44)

class ItemBasedCF(object):
    ''' TopN recommendation - Item Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_item = 100
        self.n_rec_item = 30

        self.item_sim_mat = {}
        self.item_popular = {}
        self.item_count = 0

        print('Similar item number = %d' % self.n_sim_item)
        print('Recommended item number = %d' % self.n_rec_item)

    def generate_dataset(self, train,test=None, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0
        if test is None:# 随机分配验证机实验 
            print('算法尝试！')
            for line in train.iterrows():
                user, item, rating = line[1][0],line[1][1],1
                # split the data by pivot
                if random.random() < pivot:
                    self.trainset.setdefault(user, {})
                    self.trainset[user][item] = int(rating)
                    trainset_len += 1
                else:
                    self.testset.setdefault(user, {})
                    self.testset[user][item] = int(rating)
                    testset_len += 1
        else:#真正预测
            print('预测尝试！')
            for line in train.iterrows():
                user, item, rating = line[1][0],line[1][1],1
                # split the data by pivot
                self.trainset.setdefault(user, {})
                self.trainset[user][item] = int(rating)
                trainset_len += 1
            del user,item,rating
            for line in test.iterrows():
                user, item, rating = line[1][0],line[1][1],1
                self.testset.setdefault(user, {})
                self.testset[user][item] = int(rating)
                testset_len += 1
            
        print ('split training set and test set succ')
        print ('train set = %s' % trainset_len)
        print ('test set = %s' % testset_len)

    def calc_item_sim(self):
        ''' calculate item similarity matrix '''
        print('counting items number and popularity...')

        for user, items in self.trainset.items():
            for item in items:
                # count item popularity
                if item not in self.item_popular:
                    self.item_popular[item] = 0
                self.item_popular[item] += 1

        print('count items number and popularity succ')

        # save the total number of items
        self.item_count = len(self.item_popular)
        print('total item number = %d' % self.item_count)

        # count co-rated users between items
        itemsim_mat = self.item_sim_mat
        print('building co-rated users matrix...')

        for user, items in self.trainset.items():
            for m1 in items:
                itemsim_mat.setdefault(m1, defaultdict(int))
                for m2 in items:
                    if m1 == m2:
                        continue
                    itemsim_mat[m1][m2] += 1

        print('build co-rated users matrix succ')

        # calculate similarity matrix
        print('calculating item similarity matrix...')
        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_items in itemsim_mat.items():
            for m2, count in related_items.items():
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.item_popular[m1] * self.item_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating item similarity factor(%d)' %
                          simfactor_count)

        print('calculate item similarity matrix(similarity factor) succ')
        print('Total similarity factor number = %d' % simfactor_count)

    def recommend(self, user,predict=False):
        ''' Find K similar items and recommend N items. '''
        K = self.n_sim_item
        N = self.n_rec_item
        rank = {}
        if not predict:
            watched_items = self.trainset[user]
        else:
            watched_items = self.testset[user]

        for item, rating in watched_items.items():
            if itemcf.item_sim_mat.get(item,None) is None:
                continue
            for related_item, similarity_factor in sorted(self.item_sim_mat[item].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:
                if related_item in watched_items:
                    continue
                rank.setdefault(related_item, 0)
                rank[related_item] += similarity_factor * rating
        # return the N best items
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('Evaluation start...')

        N = self.n_rec_item
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_items = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i)
            test_items = self.testset.get(user, {})
            rec_items = self.recommend(user)
            for item, _ in rec_items:
                if item in test_items:
                    hit += 1
                all_rec_items.add(item)
                popular_sum += math.log(1 + self.item_popular[item])
            rec_count += N
            test_count += len(test_items)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_items) / (1.0 * self.item_count)
        popularity = popular_sum / (1.0 * rec_count)

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
               (precision, recall, coverage, popularity))
    def predict(self):
        print('Predict start...')

        N = self.n_rec_item
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_items = set()
        # varables for popularity
        popular_sum = 0
        predict_result=[]
        for i, user in enumerate(self.testset):
            if i % 500 == 0:
                print ('recommended for %d users' % i)
            rec_items = self.recommend(user,predict=True)
            predict_result.append((user,rec_items))
        print('Predict end...')
        return predict_result
```

```
itemcf = ItemBasedCF()
```

算法尝试

```python
itemcf.generate_dataset(train[['buyer_admin_id','item_id']][:10000])#随机选择1W个交易记录测试
```

```
itemcf.calc_item_sim()


itemcf.evaluate()
```

预测尝试

```python
itemcf.generate_dataset(train[['buyer_admin_id','item_id']],test[['buyer_admin_id','item_id']])
```

```
itemcf.calc_item_sim()
```

```
output = open('itemcf.pkl', 'wb')
pickle.dump(itemcf, output)
output.close()
```

```
result = itemcf.predict()
```

```python
result_df = pd.DataFrame(result,columns=['buyer_admin_id','rec_items'])
result_df['rec_items'] = result_df.rec_items.apply(lambda x:[i[0] for i in x])
```

```
result_df['len'].value_counts()[:5]
```

使用活跃商品填补

```python
popularity_check = itemcf.item_popular

popularity_check_sorted = sorted(popularity_check.items(), lambda x, y: cmp(x[1], y[1]), reverse=True) 



for i in range(30):
    result_df['predict '+str(i+1)] = result_df.rec_items.apply(lambda x:x[i] if len(x) >i else random.choice( popularity_check_sorted [:100])[0])


result_df[result_df['len']<20].head(2)

result_df[result_df.buyer_admin_id.isin(test.buyer_admin_id.unique())].drop(['rec_items','len'],axis=1).to_csv('username.csv',index=False,header=False)
```



SVD 
====

**简介**

SVD 全称：Singular Value Decomposition。SVD 是一种提取信息的强大工具，它提供了一种非常便捷的矩阵分解方式，能够发现数据中十分有意思的潜在模式。

主要应用

- 推荐系统 (Recommender system)，可以说是最有价值的应用点；



**线性变换**

在做 SVD 推导之前，先了解一下线性变换，以 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation.svg) 的线性变换矩阵为例，先看简单的对角矩阵：

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875973895.svg)

从集合上讲， ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974128.svg) 是将二维平面上的点 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875973827.svg) 经过线性变换到另一个点的变换矩阵，如下所示：

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875973891.svg)

该变换的几何效果是，变换后的平面沿着 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974272.svg) 水平方向进行了3倍拉伸，垂直方向没有发生变化。

**SVD原理**

该部分的推导从几何层面上去理解二维的SVD，总体的思想是：借助 SVD 可以将一个相互垂直的网格 (orthogonal grid) 变换到另外一个互相垂直的网格。

可以通过二维空间中的向量来描述这件事情。

首先，选择两个互相正交的单位向量 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875973894.svg) 和 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875973897.svg) （也可称为一组正交基）。

![[公式]](https://www.zhihu.com/equation?tex=M) 是一个变换矩阵。

向量 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875973893.svg) 也是一组正交向量（也就是 ![[公式]](https://www.zhihu.com/equation?tex=v_1) 和 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875973897.svg) 经过 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974128.svg) 变换得到的）。

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974124.svg) ， ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974127.svg) 分别是 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974134.svg), ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974130.svg) 的单位向量（即另一组正交基），且有：

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974133.svg)

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974131.svg)

则， ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974138.svg) 分别为 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974139.svg) 的模（**也称为** ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974128.svg) **的奇异值**）。

设任意向量 ![[公式]](https://www.zhihu.com/equation?tex=x) ，有：

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974274.svg)

例如，当 ![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974275.svg) 时， ![[公式]](https://www.zhihu.com/equation?tex=x%3D%5Cleft%28+%5Cbegin%7Bbmatrix%7D+1+%5C%5C+0+%5Cend%7Bbmatrix%7D+%5Cbegin%7Bbmatrix%7D+3+%26+2+%5Cend%7Bbmatrix%7D+%5Cright%29+%5Cbegin%7Bbmatrix%7D+1+%5C%5C+0+%5Cend%7Bbmatrix%7D+%2B+%5Cleft%28+%5Cbegin%7Bbmatrix%7D+0+%5C%5C+1+%5Cend%7Bbmatrix%7D+%5Cbegin%7Bbmatrix%7D+3+%26+2+%5Cend%7Bbmatrix%7D+%5Cright%29+%5Cbegin%7Bbmatrix%7D+0+%5C%5C+1+%5Cend%7Bbmatrix%7D) .

那么，可得：

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974278.svg)

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974281.svg)

根据线代知识，向量的内积可用向量的转置来表示：

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974280.svg) ，则有：

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974420.svg)

两边去掉 ![[公式]](https://www.zhihu.com/equation?tex=x) ，得：

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974424.svg)

将下标相同的向量合并起来，则该式可通用地表达为：

![[公式]](每个Python数据分析师不可不知的推荐算法.assets/equation-1566875974426.svg)



```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tfcf.metrics import mae  # 需要安装这个包；最上面有link
from tfcf.metrics import rmse
from tfcf.datasets import ml1m
from tfcf.config import Config
from svd import SVD
from tfcf.models.svdpp import SVDPP
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
```

```python
import numpy as np
import tensorflow as tf

try:
    from tensorflow.keras import utils
except:
    from tensorflow.contrib.keras import utils
from tfcf.models.model_base import BaseModel
from tfcf.utils.data_utils import BatchGenerator
from tfcf.metrics import mae
from tfcf.metrics import rmse

class SVD(BaseModel):
    """Collaborative filtering model based on SVD algorithm.
    """

    def __init__(self, config, sess):
        super(SVD, self).__init__(config)
        self._sess = sess

    def _create_placeholders(self):
        """Returns the placeholders.
        """
        with tf.variable_scope('placeholder'):
            users = tf.placeholder(tf.int32, shape=[None, ], name='users')
            items = tf.placeholder(tf.int32, shape=[None, ], name='items')
            ratings = tf.placeholder(
                tf.float32, shape=[None, ], name='ratings')

        return users, items, ratings

    def _create_constants(self, mu):
        """Returns the constants.
        """
        with tf.variable_scope('constant'):
            _mu = tf.constant(mu, shape=[], dtype=tf.float32)

        return _mu

    def _create_user_terms(self, users):
        """Returns the tensors related to users.
        """
        num_users = self.num_users
        num_factors = self.num_factors

        with tf.variable_scope('user',reuse=tf.AUTO_REUSE):
            self.user_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_users, num_factors],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_p_u))

            self.user_bias = tf.get_variable(
                name='bias',
                shape=[num_users, ],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_b_u))

            p_u = tf.nn.embedding_lookup(
                self.user_embeddings,
                users,
                name='p_u')

            b_u = tf.nn.embedding_lookup(
                self.user_bias,
                users,
                name='b_u')

        return p_u, b_u

    def _create_item_terms(self, items):
        """Returns the tensors related to items.
        """
        num_items = self.num_items
        num_factors = self.num_factors

        with tf.variable_scope('item',reuse=tf.AUTO_REUSE):
            self.item_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_items, num_factors],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_q_i))

            self.item_bias = tf.get_variable(
                name='bias',
                shape=[num_items, ],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_b_i))

            q_i = tf.nn.embedding_lookup(
                self.item_embeddings,
                items,
                name='q_i')

            b_i = tf.nn.embedding_lookup(
                self.item_bias,
                items,
                name='b_i')

        return q_i, b_i
    def _get_item_terms(self):
        """Returns the tensors related to items.
        """
        import pickle
        saver = tf.train.Saver()
        item_embeddings = self.item_embeddings.eval()
        saver_path = saver.save(self._sess, './')
        print("saver path: ",saver_path)
        with open('./item_embeddings.pkl', 'wb') as fw:
            pickle.dump({'embeddings': item_embeddings}, fw)

    def _create_prediction(self, mu, b_u, b_i, p_u, q_i):
        """Returns the tensor of prediction.

           Note that the prediction 
            r_hat = \mu + b_u + b_i + p_u * q_i
        """
        with tf.variable_scope('prediction'):
            pred = tf.reduce_sum(
                tf.multiply(p_u, q_i),
                axis=1)

            pred = tf.add_n([b_u, b_i, pred])

            pred = tf.add(pred, mu, name='pred')

        return pred

    def _create_loss(self, pred, ratings):
        """Returns the L2 loss of the difference between
            ground truths and predictions.

           The formula is here:
            L2 = sum((r - r_hat) ** 2) / 2
        """
        with tf.variable_scope('loss'):
            loss = tf.nn.l2_loss(tf.subtract(ratings, pred), name='loss')

        return loss

    def _create_optimizer(self, loss):
        """Returns the optimizer.

           The objective function is defined as the sum of
            loss and regularizers' losses.
        """
        with tf.variable_scope('optimizer'):
            objective = tf.add(
                loss,
                tf.add_n(tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)),
                name='objective')

            try:
                optimizer = tf.contrib.keras.optimizers.Nadam(
                ).minimize(objective, name='optimizer')
            except:
                optimizer = tf.train.AdamOptimizer().minimize(objective, name='optimizer')

        return optimizer

    def _build_graph(self, mu):
        _mu = self._create_constants(mu)

        self._users, self._items, self._ratings = self._create_placeholders()

        p_u, b_u = self._create_user_terms(self._users)
        q_i, b_i = self._create_item_terms(self._items)

        self._pred = self._create_prediction(_mu, b_u, b_i, p_u, q_i)

        loss = self._create_loss(self._ratings, self._pred)

        self._optimizer = self._create_optimizer(loss)

        self._built = True

    def _run_train(self, x, y, epochs, batch_size, validation_data):
        train_gen = BatchGenerator(x, y, batch_size)
        steps_per_epoch = np.ceil(train_gen.length / batch_size).astype(int)

        self._sess.run(tf.global_variables_initializer())

        for e in range(1, epochs + 1):
            print('Epoch {}/{}'.format(e, epochs))

            pbar = utils.Progbar(steps_per_epoch)

            for step, batch in enumerate(train_gen.next(), 1):
                users = batch[0][:, 0]
                items = batch[0][:, 1]
                ratings = batch[1]

                self._sess.run(
                    self._optimizer,
                    feed_dict={
                        self._users: users,
                        self._items: items,
                        self._ratings: ratings
                    })

                pred = self.predict(batch[0])

                update_values = [
                    ('rmse', rmse(ratings, pred)),
                    ('mae', mae(ratings, pred))
                ]

                if validation_data is not None and step == steps_per_epoch:
                    valid_x, valid_y = validation_data
                    valid_pred = self.predict(valid_x)

                    update_values += [
                        ('val_rmse', rmse(valid_y, valid_pred)),
                        ('val_mae', mae(valid_y, valid_pred))
                    ]

                pbar.update(step, values=update_values)

    def train(self, x, y, epochs=100, batch_size=1024, validation_data=None):

        if x.shape[0] != y.shape[0] or x.shape[1] != 2:
            raise ValueError('The shape of x should be (samples, 2) and '
                             'the shape of y should be (samples, 1).')

        if not self._built:
            self._build_graph(np.mean(y))

        self._run_train(x, y, epochs, batch_size, validation_data)
        self._get_item_terms()

    def predict(self, x):
        if not self._built:
            raise RunTimeError('The model must be trained '
                               'before prediction.')

        if x.shape[1] != 2:
            raise ValueError('The shape of x should be '
                             '(samples, 2)')

        pred = self._sess.run(
            self._pred,
            feed_dict={
                self._users: x[:, 0],
                self._items: x[:, 1]
            })

        pred = pred.clip(min=self.min_value, max=self.max_value)

        return pred

```

```python
train_counts = train.groupby(['buyer_admin_id','item_id'])['count'].sum().reset_index()


train_counts['count_ratio']=train_counts['count']/1271.0


train_counts.head()


```

```python
train_counts.buyer_admin_id.min(),train_counts.buyer_admin_id.max() # 次数对不上embedding，使用labelencoder规范化
```

```python
from sklearn import preprocessing
le_buyer = preprocessing.LabelEncoder()
le_buyer.fit(train_counts.buyer_admin_id.unique()) 
le_item = preprocessing.LabelEncoder()
le_item.fit(train_counts.item_id.unique()) 
```

```python
train_counts['buyer'] = le_buyer.transform(train_counts.buyer_admin_id.values)
train_counts['item'] = le_item.transform(train_counts.item_id.values)


train_counts['buyer'] = train_counts['buyer']+1
train_counts['item'] = train_counts['item']+1

train_counts.head()


x,y=train_counts[['buyer','item']].values,train_counts.count_ratio.values
```

算法尝试

购买次数越多的推荐的概率越高，来学习item的embedding 层，svd的pq

```python
# Note that x is a 2D numpy array, 
# x[i, :] contains the user-item pair, and y[i] is the corresponding rating.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

config = Config()
config.num_users = np.max(x[:, 0]) + 1
config.num_items = np.max(x[:, 1]) + 1
config.min_value = np.min(y)
config.max_value = np.max(y)

with tf.Session() as sess:
    # For SVD++ algorithm, if `dual` is True, then the dual term of items' 
    # implicit feedback will be added into the original SVD++ algorithm.
    # model = SVDPP(config, sess, dual=False)
    # model = SVDPP(config, sess, dual=True)
    model = SVD(config, sess)# 给模型加了保存embedding的一个功能
    model.train(x_train, y_train, validation_data=(
        x_test, y_test), epochs=2, batch_size=8092)
        
    y_pred = model.predict(x_test)
    print('rmse: {}, mae: {}'.format(rmse(y_test, y_pred), mae(y_test, y_pred)))
        
    # Save model
    model.save_model('model/')
```

预测尝试

购买次数越多的推荐的概率越高，来学习item的embedding 层，svd的pq

- 使用knn找到最相似的item

```python
import pickle
with open('./item_embeddings.pkl', 'rb') as fr:
    data = pickle.load(fr)
    final_embeddings = data['embeddings']


test_counts = test[test.item_id.isin(train.item_id.unique())].groupby(['buyer_admin_id','item_id']).agg({'count':'sum','irank':'max'}).reset_index()
test_counts['item'] = le_item.transform(test_counts.item_id.values)


test_counts.head()



from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=31, algorithm='ball_tree').fit(final_embeddings)



test_final_embeddings=[]
for line in test_counts.iterrows():
    test_final_embeddings.append(final_embeddings[line[1]['item']])


len(test_final_embeddings)



# 比较费时间
distances, indices = nbrs.kneighbors(test_final_embeddings) # 已经按照距离排好序了



indices_inversed = le_item.inverse_transform(indices)



indices_inversed_str = [str(li) for li in indices_inversed]


indices_inversed[:,0]



for i in range(indices_inversed.shape[1]):
    test_counts['recomend_'+str(i)] = indices_inversed[:,i]



test_counts.head()


test_counts.to_csv('test_counts.csv',index=False,encoding='gbk')

```

- 从最优购买的前几个item中选择30个；第一个选15个剩下3个都选5个的思路

```
test_counts = pd.read_csv('test_counts.csv')

test_counts.head()
```

```python
test_counts.groupby('buyer_admin_id').apply(lambda t: t[t.irank==t.irank.min()]).reset_index(drop=True).drop(['item_id','count','irank','item','rec_items_labled','recomend_0'],axis=1).to_csv('username_svd.csv',index=False,header=False,encoding='gbk')
```

