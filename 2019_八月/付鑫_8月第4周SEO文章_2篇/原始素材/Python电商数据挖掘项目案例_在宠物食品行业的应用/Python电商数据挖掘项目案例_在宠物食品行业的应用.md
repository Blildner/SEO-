Python电商数据挖掘项目案例——宠物食品行业应用_Part1
==================================================

[TOC]

业务背景——宠物食品行业
======================



**产业概述**
------------

**宠物一般是指家庭饲养的、作为伴侣动物的犬和猫等。**

根据海通证券研究报告，2015年美国饲养宠物的家庭中，宠物狗、宠物猫占比达到75%，中国为81%。

### 宠物食品

**宠物食品是专门为宠物提供的食品，介于人类食品与传统畜禽饲料之间，其作用主要是为各种宠物提供最基础的生命保证、生长发育和健康所需的营养物质，具有营养全面、消化吸收率高、配方科学、饲喂方便以及可预防某些疾病等优点。**

**按功能划分，宠物食品主要可分为**

- **宠物主食、**
- **宠物零食和**
- **宠物保健品**，

其中宠物主食占采食量的70%以上。

![1566803086499](Python电商数据挖掘项目案例_在宠物食品行业的应用.assets/1566803086499.png)

*资料来源：佩蒂股份招股说明书*



### 宠物产业链

**从产业链角度出发，宠物食品位于宠物行业上游，在宠物市场中占比最大。**根据有庞研究院及天风证券研究所报告，2016年宠物食品在我国宠物市场占比约37.5%，在美国为38.26%。除宠物食品，宠物产业链还包括宠物饲养、宠物食品、宠物用品、宠物医疗、宠物美容、宠物培训、宠物保险以及宠物善终等。

![宠物行业](Python电商数据挖掘项目案例_在宠物食品行业的应用.assets/宠物行业.png)

*资料来源：安信证券研究中心*



### 宠物食品市场规模

**人均收入提升带来的消费升级，老龄化比率上升增加了老年人对宠物的需求，我国宠物食品市场目前正处于高速增长期**。

根据前瞻产业研究院的报告，

2008年-2015年我国宠物食品市场规模复合增长率超过30%；

2016年我国宠物食品的市场规模约457亿元, 同比增长38.9%，宠物零食市场规模约90亿元，同比增长42.8%。

**较于发达国家，我国宠物食品产业仍旧滞后，家庭宠物保有量、宠物的年消费支出较低，未来仍然具有广阔的发展空间**。

天风证券预计2020年我国宠物食品市场规模将达到 1160亿元，较2016年增长137.7%，宠物零食的市场规模将达到270亿元，较2016年增长200%。

![增长率2](Python电商数据挖掘项目案例_在宠物食品行业的应用.assets/增长率2-1566806402644.png)

*资料来源：有宠研究院*



项目数据分析
============

```python
import os
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import jieba

import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize = (12,8))

%matplotlib inline
plt.rcParams['axes.unicode_minus'] = False

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['Simhei']
from wordcloud import WordCloud,ImageColorGenerator
import imageio
import snownlp
from snownlp import SnowNLP
```

```python
os.getcwd()
```



```
'C:\\Users\\Administrator\\a'
```





数据处理（一）
==============

**版本问题**

 wheel安装步骤 
    - 下载适合自己python版本的包：https://www.lfd.uci.edu/~gohlke/pythonlibs/  
    - pip install wheel
    - 目标文件夹的cd,pip install somewhat.whl

```python
print("pandas 版本：",pd.__version__)
print("numpy 版本：",np.__version__)
```

```
pandas 版本： 0.23.4
numpy 版本： 1.16.3
```

- 检查更新：pip list --outdated
- 更新： pip install --upgrade xxxx

pandas 0.23.4 documentation  http://pandas.pydata.org/pandas-docs/stable/index.html

销售数据概况
------------

```python
df_raw_1 = pd.read_excel('Product_details_rawdata.xlsx')  # 183 rows x 25 columns
df_raw_1.columns
```



```
Index(['item_id', 'item_name', 'TradeName', 'price', 'total_sale',
       'month_sale', 'accum_comm', 'TM_points', 'CollectCount', 'Tastes',
       'BodyType', 'ApplicablePhase', 'Brand', 'Classification', 'Breed',
       'Manufacturer', 'Weight', 'Origin', 'ManufacturerAddress',
       'RecipeTastePrescription'],
      dtype='object')
```



```python
df_raw_1.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_id</th>
      <th>item_name</th>
      <th>TradeName</th>
      <th>price</th>
      <th>total_sale</th>
      <th>month_sale</th>
      <th>accum_comm</th>
      <th>TM_points</th>
      <th>CollectCount</th>
      <th>Tastes</th>
      <th>BodyType</th>
      <th>ApplicablePhase</th>
      <th>Brand</th>
      <th>Classification</th>
      <th>Breed</th>
      <th>Manufacturer</th>
      <th>Weight</th>
      <th>Origin</th>
      <th>ManufacturerAddress</th>
      <th>RecipeTastePrescription</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19045209534</td>
      <td>Royal Canin皇家狗粮 德牧幼犬粮AGS30 12KG 大型犬狗粮28省包邮</td>
      <td>德国牧羊犬幼犬专用粮 12kg</td>
      <td>650.0</td>
      <td>2046</td>
      <td>39</td>
      <td>418</td>
      <td>325</td>
      <td>348</td>
      <td>其他</td>
      <td>通用型</td>
      <td>幼犬</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>德国牧羊犬</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>上海</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>547144906363</td>
      <td>Royal Canin皇家狗粮 柴犬幼犬专用粮SIJ29/3KG 犬主粮狗粮</td>
      <td>日本柴犬幼犬 3000g</td>
      <td>285.0</td>
      <td>560</td>
      <td>39</td>
      <td>132</td>
      <td>142</td>
      <td>200</td>
      <td>其他</td>
      <td>NaN</td>
      <td>幼犬</td>
      <td>ROYAL CANIN/皇家</td>
      <td>NaN</td>
      <td>日本柴犬</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26255560615</td>
      <td>皇家狗粮 大型犬奶糕MAS30 4KG哺乳孕期犬离乳期幼犬牧羊阿拉斯加</td>
      <td>(大型犬)离乳期奶糕 4kg</td>
      <td>301.0</td>
      <td>1571</td>
      <td>27</td>
      <td>390</td>
      <td>150</td>
      <td>554</td>
      <td>其他</td>
      <td>大型犬</td>
      <td>离乳期</td>
      <td>ROYAL CANIN/皇家</td>
      <td>奶糕</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>4000</td>
      <td>中国</td>
      <td>上海</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39792870853</td>
      <td>Royal Canin皇家狗粮 迷你雪纳瑞成犬粮SNZ25/3KG 犬主粮</td>
      <td>雪纳瑞成犬 3000g</td>
      <td>260.0</td>
      <td>2113</td>
      <td>32</td>
      <td>281</td>
      <td>130</td>
      <td>379</td>
      <td>其他</td>
      <td>NaN</td>
      <td>成犬</td>
      <td>ROYAL CANIN/皇家</td>
      <td>NaN</td>
      <td>雪纳瑞</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海奉贤区肖南路475号</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>547165690913</td>
      <td>Royal Canin皇家狗粮 拉布拉多幼犬粮ALR33 3KG 大型犬全犬种热卖</td>
      <td>拉布拉多幼犬 3000g</td>
      <td>210.0</td>
      <td>643</td>
      <td>48</td>
      <td>173</td>
      <td>105</td>
      <td>318</td>
      <td>其他</td>
      <td>NaN</td>
      <td>幼犬</td>
      <td>ROYAL CANIN/皇家</td>
      <td>NaN</td>
      <td>拉布拉多</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

</div>



```python
len(df_raw_1)
```



```
183
```



```python
df_raw_1.describe()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_id</th>
      <th>price</th>
      <th>total_sale</th>
      <th>month_sale</th>
      <th>accum_comm</th>
      <th>TM_points</th>
      <th>CollectCount</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.830000e+02</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.669680e+11</td>
      <td>307.403279</td>
      <td>11136.404372</td>
      <td>407.229508</td>
      <td>2648.890710</td>
      <td>153.508197</td>
      <td>1879.065574</td>
      <td>5344.699454</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.556689e+11</td>
      <td>257.744147</td>
      <td>31369.026185</td>
      <td>1068.822091</td>
      <td>6659.411861</td>
      <td>128.909154</td>
      <td>4007.666013</td>
      <td>4927.903706</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.562877e+10</td>
      <td>9.900000</td>
      <td>15.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>13.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.669303e+10</td>
      <td>135.500000</td>
      <td>1064.500000</td>
      <td>39.000000</td>
      <td>267.500000</td>
      <td>67.500000</td>
      <td>248.500000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.388117e+10</td>
      <td>225.000000</td>
      <td>2883.000000</td>
      <td>149.000000</td>
      <td>736.000000</td>
      <td>112.000000</td>
      <td>584.000000</td>
      <td>3000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.364667e+11</td>
      <td>460.000000</td>
      <td>7704.500000</td>
      <td>322.500000</td>
      <td>2090.000000</td>
      <td>230.000000</td>
      <td>1458.500000</td>
      <td>7500.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.649716e+11</td>
      <td>1300.000000</td>
      <td>337524.000000</td>
      <td>9931.000000</td>
      <td>56434.000000</td>
      <td>650.000000</td>
      <td>34370.000000</td>
      <td>30000.000000</td>
    </tr>
  </tbody>
</table>
</div>



describe方法用于一次性产生多个汇总统计,默认对数值型数据.对于非数值型数据,describe会产生出另外一种汇总统计.

```python
df_raw_1.item_name.describe()
```



```
count                                   183
unique                                  183
top       宝路狗粮狗主粮泰迪比熊贵宾中型小型通用型成犬牛肉味15斤7.5kg
freq                                      1
Name: item_name, dtype: object
```



共有183种不同的产品

### 品牌

```python
df_raw_1.Brand.describe()
```



```
count                183
unique                 5
top       ROYAL CANIN/皇家
freq                  91
Name: Brand, dtype: object
```



```python
df_raw_1.Brand.value_counts(dropna=False)
```



```
ROYAL CANIN/皇家       91
Pedigree/宝路          38
Nature Bridge/比瑞吉    28
Fish4Dogs/海洋之星       24
Chappi/佳贝             2
Name: Brand, dtype: int64
```



### 口味

```python
df_raw_1.Tastes.describe()
```



```
count     183
unique      5
top        其他
freq      130
Name: Tastes, dtype: object
```



```python
df_raw_1.Tastes.unique()
```



```
array(['其他', '牛肉味', '鸡肉味', '鱼肉味', '深海鱼味'], dtype=object)
```



```python
df_raw_1.Tastes.value_counts(dropna=False)
```



```
其他      130
鱼肉味      25
牛肉味      17
鸡肉味       9
深海鱼味      2
Name: Tastes, dtype: int64
```



### 适用体型

```python
df_raw_1.BodyType.describe()
```



```
count     137
unique      6
top       通用型
freq       47
Name: BodyType, dtype: object
```



```python
df_raw_1.BodyType.unique()
```



```
array(['通用型', nan, '大型犬', '小型犬', '中型犬', '中小型犬', '中大型犬'], dtype=object)
```



```python
df_raw_1.BodyType.value_counts(dropna=False)
```



```
通用型     47
小型犬     46
NaN     46
大型犬     20
中小型犬    16
中型犬      6
中大型犬     2
Name: BodyType, dtype: int64
```



### 适用年龄

```python
df_raw_1.ApplicablePhase.describe()
```



```
count     183
unique      6
top        成犬
freq       92
Name: ApplicablePhase, dtype: object
```



```python
df_raw_1.ApplicablePhase.value_counts(dropna=False)
```



```
成犬                               92
幼犬                               55
老年犬                              13
全犬期                              12
离乳期                               7
使用于怀孕42天起的母犬、哺乳期母犬及2月龄以下离乳期幼犬     4
Name: ApplicablePhase, dtype: int64
```



### 狗粮种类

```python
df_raw_1.Classification.describe()
```



```
count     148
unique      3
top        犬粮
freq      104
Name: Classification, dtype: object
```



```python
df_raw_1.Classification.value_counts(dropna=False)
```



```
犬粮     104
专用粮     36
NaN     35
奶糕       8
Name: Classification, dtype: int64
```



### 适用品种

```python
df_raw_1.Breed.describe()
```



```
count     156
unique     14
top       通用型
freq      106
Name: Breed, dtype: object
```



```python
df_raw_1.Breed.value_counts(dropna=False)
```



```
通用型       106
NaN        27
贵宾/泰迪      15
约克夏梗        4
日本柴犬        4
金毛          4
拉布拉多        4
斗牛犬         4
雪纳瑞         3
比熊          3
德国牧羊犬       3
吉娃娃         2
可卡          2
博美          1
西部高地白梗      1
Name: Breed, dtype: int64
```



### 功能性配方

```python
df_raw_1.RecipeTastePrescription.describe()
```



```
count     79
unique    48
top        无
freq       6
Name: RecipeTastePrescription, dtype: object
```



### 原产国

```python
df_raw_1.Origin.describe()
```



```
count     183
unique      3
top        中国
freq      169
Name: Origin, dtype: object
```



```python
df_raw_1.Origin.value_counts(dropna=False)
```



```
中国     169
比利时     10
其他       4
Name: Origin, dtype: int64
```



### 厂商地址

```python
df_raw_1.ManufacturerAddress.describe()
```



```
count     183
unique     12
top        上海
freq       79
Name: ManufacturerAddress, dtype: object
```



```python
df_raw_1.ManufacturerAddress.value_counts(dropna=False)
```



```
上海                                    79
北京                                    35
上海市金山区                                20
Terdonkkaai 16, 9042 Gent, Belgium    10
山东省聊城市经济开发区牡丹江路8号                     10
上海奉贤区肖南路475号                           8
上海市金山区亭卫公路                             7
中国                                     5
Fish4Dogs.Ltd                          4
上海上海奉贤区肖南路475号                         2
皇誉宠物食品（上海）有限公司                         2
上海金山区                                  1
Name: ManufacturerAddress, dtype: int64
```



数据概括总结：

- 一共有183种不同产品
- 分别来自于4种主流品牌
- 商品属性信息有：商品名，种类，适用品种，适用体型，适用年龄，净重，功能性配方
- 商品销售信息有：价格，总销售量，月销售量，累计评论，收藏量，

```python
import jieba
```

```python
# import jieba.posseg as psg
```

```python
jieba.__version__  # 0.38
```



```
'0.39'
```



对最详细的产品名进行分词处理

```python
b= jieba.cut(df_raw_1.item_name[1])
for i in b:
    print(i)

```

```
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\ADMINI~1\AppData\Local\Temp\jieba.cache
Loading model cost 1.632 seconds.
Prefix dict has been built succesfully.
```

```
Royal
 
Canin
皇家
狗
粮
 
柴犬
幼犬
专用
粮
SIJ29
/
3KG
 
犬
主粮
狗
粮
```

存在专有名词，默认无法分割正确时，加入特定词语list

```python
 jieba.load_userdict('AddWords.txt')
```

```python
c= jieba.lcut(df_raw_1.item_name[1])
c
```



```
['Royal',
 ' ',
 'Canin',
 '皇家',
 '狗粮',
 ' ',
 '柴犬',
 '幼犬',
 '专用粮',
 'SIJ29',
 '/',
 '3KG',
 ' ',
 '犬主粮',
 '狗粮']
```



对所有item name 进行分词，添加到源数据中

```python
item_cut=[]
for i in df_raw_1.item_name:
    j=jieba.lcut(i)
    item_cut.append(j)
```

```python
df_raw_1['item_name_cut'] = item_cut
```

```python
df_raw_1[['item_name','item_name_cut']]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_name</th>
      <th>item_name_cut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Royal Canin皇家狗粮 德牧幼犬粮AGS30 12KG 大型犬狗粮28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 德牧, 幼犬粮, AGS30,  ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Royal Canin皇家狗粮 柴犬幼犬专用粮SIJ29/3KG 犬主粮狗粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 幼犬, 专用粮, SIJ2...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>皇家狗粮 大型犬奶糕MAS30 4KG哺乳孕期犬离乳期幼犬牧羊阿拉斯加</td>
      <td>[皇家, 狗粮,  , 大型犬, 奶糕, MAS30,  , 4KG, 哺乳, 孕期, 犬离...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Royal Canin皇家狗粮 迷你雪纳瑞成犬粮SNZ25/3KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 迷你, 雪纳瑞, 成犬粮, SNZ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Royal Canin皇家狗粮 拉布拉多幼犬粮ALR33 3KG 大型犬全犬种热卖</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 拉布拉多, 幼犬粮, ALR33,...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Royal Canin皇家狗粮 小型犬老年犬狗粮SPR27/0.8kg公斤  犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 小型犬, 老年犬, 狗粮, SPR...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Royal Canin皇家狗粮 大型犬幼犬粮MAJ30/15KG  犬主粮28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 幼犬粮, MAJ30, ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Royal Canin皇家狗粮 约克夏成犬专用粮PRY28/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 约克夏, 成犬, 专用粮, PRY...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>皇家狗粮 小型犬成犬通用型PR27 2KG 博美迷你腊肠京巴八哥全犬种</td>
      <td>[皇家, 狗粮,  , 小型犬, 成犬, 通用型, PR27,  , 2KG,  , 博美,...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Royal Canin皇家狗粮 法国斗牛犬幼犬粮FBJ30/3KG 法斗粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 法国, 斗牛犬, 幼犬粮, FBJ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Royal Canin皇家狗粮 小型犬幼犬粮MIJ31 8KG 犬主粮28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 小型犬, 幼犬粮, MIJ31, ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>皇家繁育期母犬 离乳期幼犬狗粮中型犬奶糕MES30 4KG 松狮哈士奇</td>
      <td>[皇家, 繁育, 期母, 犬,  , 离乳期, 幼犬, 狗粮, 中型犬, 奶糕, MES30...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>皇家狗粮 吉娃娃成犬专用粮食C28 1.5KG*2  小型犬 28省包邮</td>
      <td>[皇家, 狗粮,  , 吉娃娃, 成犬, 专用, 粮食, C28,  , 1.5, KG, ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>皇家狗粮 小型犬成犬狗粮 通用型狗粮均衡营养减少牙石PR27/0.8KG</td>
      <td>[皇家, 狗粮,  , 小型犬, 成犬, 狗粮,  , 通用型, 狗粮, 均衡, 营养, 减...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>皇家小型犬幼犬粮MIJ31 2KG博美西施杜宾腊肠犬狗粮通用型全犬种</td>
      <td>[皇家, 小型犬, 幼犬粮, MIJ31,  , 2KG, 博美, 西施, 杜宾, 腊肠犬,...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Royal Canin皇家狗粮 泰迪/贵宾成犬粮PD30/7.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 泰迪, /, 贵宾, 成犬粮, P...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>皇家狗粮中型犬成犬粮M25 4KG哈士奇松狮萨摩耶粮柯基牛头梗热卖</td>
      <td>[皇家, 狗粮, 中型犬, 成犬粮, M25,  , 4KG, 哈士奇, 松狮, 萨摩耶, ...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Roayl Canin皇家狗粮 贵宾幼犬粮 泰迪专用粮APD33/0.5kg 犬主粮</td>
      <td>[Roayl,  , Canin, 皇家, 狗粮,  , 贵宾, 幼犬粮,  , 泰迪, 专...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>皇家官方旗舰店 贵宾泰迪狗粮8岁以上老年犬PDA26 3KG热卖新品</td>
      <td>[皇家, 官方, 旗舰店,  , 贵宾, 泰迪, 狗粮, 8, 岁, 以上, 老年犬, PD...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>皇家狗粮小型犬8岁以上成犬粮SPR27 4KG*2  博美犬茶杯 28省包邮</td>
      <td>[皇家, 狗粮, 小型犬, 8, 岁, 以上, 成犬粮, SPR27,  , 4KG, *,...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Royal Canin皇家狗粮 贵宾/泰迪成犬粮PD30/3KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 贵宾, /, 泰迪, 成犬粮, P...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Royal Canin皇家狗粮 大型犬成犬粮GR26/15KG 犬主粮28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 成犬粮, GR26, /...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>皇家官方旗舰店 狗粮比熊犬成犬粮BF29 3KG主粮热卖新品小型犬</td>
      <td>[皇家, 官方, 旗舰店,  , 狗粮, 比熊犬, 成犬粮, BF29,  , 3KG, 主...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>皇家狗粮 中型犬奶糕MES30 4KG*2繁育母犬离乳幼柯基萨摩28省包邮</td>
      <td>[皇家, 狗粮,  , 中型犬, 奶糕, MES30,  , 4KG, *, 2, 繁育, ...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Royal Canin皇家狗粮 贵宾成犬专用粮PD30/0.5KG 泰迪犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 贵宾, 成犬, 专用粮, PD30...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Royal Canin皇家狗粮 大型犬成犬粮GR26/4KG*4包犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 成犬粮, GR26, /...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Royal Canin皇家狗粮 金毛幼犬粮AGR29 3.5KG 大型犬狗粮精品热卖</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 金毛, 幼犬粮, AGR29,  ...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>皇家小型犬幼犬粮通用型狗粮 MIJ31 0.8KG博美八哥腊肠京巴犬热卖</td>
      <td>[皇家, 小型犬, 幼犬粮, 通用型, 狗粮,  , MIJ31,  , 0.8, KG, ...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>皇家小型犬成犬粮 通用型狗粮PR27 8KG 博美京巴全犬种28省包邮</td>
      <td>[皇家, 小型犬, 成犬粮,  , 通用型, 狗粮, PR27,  , 8KG,  , 博美...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Royal Canin皇家狗粮 约克夏成犬粮PRY28/1.5KG*2犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 约克夏, 成犬粮, PRY28, ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153</th>
      <td>比瑞吉小型成犬粮 小型犬成犬天然粮 比熊泰迪成犬比瑞吉狗粮10kg</td>
      <td>[比瑞吉, 小型, 成犬粮,  , 小型犬, 成犬, 天然粮,  , 比熊, 泰迪, 成犬,...</td>
    </tr>
    <tr>
      <th>154</th>
      <td>比瑞吉无谷六种肉全期犬粮 非五谷狗粮 无谷多肉成犬幼犬狗粮2kg</td>
      <td>[比瑞吉, 无谷, 六种, 肉, 全期犬粮,  , 非, 五谷, 狗粮,  , 无谷, 多肉...</td>
    </tr>
    <tr>
      <th>155</th>
      <td>比瑞吉狗粮 小型成犬粮通用型成犬狗粮2kg 雪纳瑞博美泰迪狗粮</td>
      <td>[比瑞吉, 狗粮,  , 小型, 成犬粮, 通用型, 成犬, 狗粮, 2kg,  , 雪纳瑞...</td>
    </tr>
    <tr>
      <th>156</th>
      <td>狗粮 比瑞吉泰迪贵宾成犬粮 泰迪狗粮成犬天然粮 比瑞吉狗粮2kg</td>
      <td>[狗粮,  , 比瑞吉, 泰迪, 贵宾, 成犬粮,  , 泰迪, 狗粮, 成犬, 天然粮, ...</td>
    </tr>
    <tr>
      <th>157</th>
      <td>比瑞吉狗粮 小型成犬粮泰迪贵宾比熊博美雪纳瑞通用天然狗粮1.5kg</td>
      <td>[比瑞吉, 狗粮,  , 小型, 成犬粮, 泰迪, 贵宾, 比熊博美, 雪纳瑞, 通用, 天...</td>
    </tr>
    <tr>
      <th>158</th>
      <td>狗粮幼犬 比瑞吉小型幼犬粮 泰迪比熊通用 室内小型全价幼犬粮2kg</td>
      <td>[狗粮, 幼犬,  , 比瑞吉, 小型, 幼犬粮,  , 泰迪, 比熊, 通用,  , 室内...</td>
    </tr>
    <tr>
      <th>159</th>
      <td>海洋之星SUPERIOR进口营养加强体重控制三文鱼狗粮泰迪通用型12kg</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 体重, 控制, 三文鱼, 狗粮, ...</td>
    </tr>
    <tr>
      <th>160</th>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼成犬狗粮泰迪通用型12kg</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, ...</td>
    </tr>
    <tr>
      <th>161</th>
      <td>海洋之星SUPERIOR进口营养加强体重控制三文鱼狗粮中大型犬12kg</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 体重, 控制, 三文鱼, 狗粮, ...</td>
    </tr>
    <tr>
      <th>162</th>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼成犬狗粮中大型犬12kg</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, ...</td>
    </tr>
    <tr>
      <th>163</th>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼幼犬狗粮中大型犬12kg</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 幼犬, 狗粮, ...</td>
    </tr>
    <tr>
      <th>164</th>
      <td>海洋之星狗粮金毛阿拉斯加大型犬幼犬无谷天然粮大颗粒犬粮12kg</td>
      <td>[海洋之星, 狗粮, 金毛, 阿拉斯加, 大型犬, 幼犬, 无谷, 天然粮, 大颗粒, 犬粮...</td>
    </tr>
    <tr>
      <th>165</th>
      <td>海洋之星SUPERIOR原装进口营养加强体重控制三文鱼配方狗粮6kg</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 体重, 控制, 三文鱼, ...</td>
    </tr>
    <tr>
      <th>166</th>
      <td>海洋之星SUPERIOR进口营养加强体重控制三文鱼狗粮通用型1.5kg</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 体重, 控制, 三文鱼, 狗粮, ...</td>
    </tr>
    <tr>
      <th>167</th>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼幼犬狗粮泰迪通用型6kg</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 幼犬, 狗粮, ...</td>
    </tr>
    <tr>
      <th>168</th>
      <td>海洋之星SUPERIOR进口营养加强三文鱼成犬狗粮泰迪通用型1.5kg</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, 泰迪, ...</td>
    </tr>
    <tr>
      <th>169</th>
      <td>大幼试吃装中大型犬幼犬金毛阿拉斯牧羊犬哈士奇狗粮三包*30g</td>
      <td>[大幼, 试吃装, 中大型犬, 幼犬, 金毛, 阿拉斯, 牧羊犬, 哈士奇, 狗粮, 三包,...</td>
    </tr>
    <tr>
      <th>170</th>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼成犬狗粮泰迪通用型6kg</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, ...</td>
    </tr>
    <tr>
      <th>171</th>
      <td>试吃装 深海鱼三文鱼成犬中大型犬试吃装大颗粒阿拉斯加三包*30g</td>
      <td>[试吃装,  , 深海鱼, 三文鱼, 成犬, 中大型犬, 试吃装, 大颗粒, 阿拉斯加, 三...</td>
    </tr>
    <tr>
      <th>172</th>
      <td>海洋之星深海鱼狗粮无谷天然粮中小型犬成犬主粮小颗粒6kg</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 中小型犬, 成, 犬主粮, 小颗粒, 6kg]</td>
    </tr>
    <tr>
      <th>173</th>
      <td>海洋之星深海鱼狗粮无谷天然粮金毛萨摩耶大型犬成犬大颗粒12kg</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 金毛, 萨摩耶, 大型犬, 成犬, 大颗粒...</td>
    </tr>
    <tr>
      <th>174</th>
      <td>海洋之星深海鱼狗粮无谷天然粮中小型犬幼犬小颗粒狗粮6kg</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 中小型犬, 幼犬, 小颗粒, 狗粮, 6kg]</td>
    </tr>
    <tr>
      <th>175</th>
      <td>海洋之星深海鱼狗粮无谷天然粮中小型犬幼犬小颗粒12kg</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 中小型犬, 幼犬, 小颗粒, 12kg]</td>
    </tr>
    <tr>
      <th>176</th>
      <td>海洋之星狗粮三文鱼无谷天然粮金毛拉布拉多大型犬成犬主粮 12kg</td>
      <td>[海洋之星, 狗粮, 三文鱼, 无谷, 天然粮, 金毛, 拉布拉多, 大型犬, 成, 犬主粮...</td>
    </tr>
    <tr>
      <th>177</th>
      <td>海洋之星深海鱼成犬狗粮无谷天然粮中小型犬成犬小颗粒1.5kg</td>
      <td>[海洋之星, 深海鱼, 成犬, 狗粮, 无谷, 天然粮, 中小型犬, 成犬, 小颗粒, 1....</td>
    </tr>
    <tr>
      <th>178</th>
      <td>海洋之星深海鱼成犬狗粮无谷天然粮中小型犬成犬狗粮12kg小颗粒</td>
      <td>[海洋之星, 深海鱼, 成犬, 狗粮, 无谷, 天然粮, 中小型犬, 成犬, 狗粮, 12k...</td>
    </tr>
    <tr>
      <th>179</th>
      <td>海洋之星狗粮试吃装泰迪比熊玩具犬幼犬成犬全犬期小颗粒三包*30g</td>
      <td>[海洋之星, 狗粮, 试吃装, 泰迪, 比熊, 玩具犬, 幼犬, 成犬, 全犬期, 小颗粒,...</td>
    </tr>
    <tr>
      <th>180</th>
      <td>海洋之星狗粮通用型天然粮中小型犬幼犬主粮泰迪博美比熊1.5kg</td>
      <td>[海洋之星, 狗粮, 通用型, 天然粮, 中小型犬, 幼犬主粮, 泰迪, 博美比, 熊, 1...</td>
    </tr>
    <tr>
      <th>181</th>
      <td>海洋之星狗粮试吃装泰迪比熊贵宾幼犬通用型小颗粒美毛三包*30g</td>
      <td>[海洋之星, 狗粮, 试吃装, 泰迪, 比熊, 贵宾, 幼犬, 通用型, 小颗粒, 美毛, ...</td>
    </tr>
    <tr>
      <th>182</th>
      <td>海洋之星三文鱼狗粮泰迪吉娃娃贵宾玩具犬狗粮无谷天然粮6kg</td>
      <td>[海洋之星, 三文鱼, 狗粮, 泰迪, 吉娃娃, 贵宾, 玩具犬, 狗粮, 无谷, 天然粮,...</td>
    </tr>
  </tbody>
</table>
<p>183 rows × 2 columns</p>

</div>



#### 种类

```python
df_raw_1.Classification.value_counts(dropna=False)
```



```
犬粮     104
专用粮     36
NaN     35
奶糕       8
Name: Classification, dtype: int64
```



存在35条空值

```python
len(df_raw_1[df_raw_1.Classification.isnull()])
```



```
35
```



```python
df_raw_1[df_raw_1.Classification.isnull()][['item_name','item_name_cut']]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_name</th>
      <th>item_name_cut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Royal Canin皇家狗粮 柴犬幼犬专用粮SIJ29/3KG 犬主粮狗粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 幼犬, 专用粮, SIJ2...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Royal Canin皇家狗粮 迷你雪纳瑞成犬粮SNZ25/3KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 迷你, 雪纳瑞, 成犬粮, SNZ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Royal Canin皇家狗粮 拉布拉多幼犬粮ALR33 3KG 大型犬全犬种热卖</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 拉布拉多, 幼犬粮, ALR33,...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>皇家狗粮 吉娃娃成犬专用粮食C28 1.5KG*2  小型犬 28省包邮</td>
      <td>[皇家, 狗粮,  , 吉娃娃, 成犬, 专用, 粮食, C28,  , 1.5, KG, ...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Royal Canin皇家狗粮 金毛幼犬粮AGR29 3.5KG 大型犬狗粮精品热卖</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 金毛, 幼犬粮, AGR29,  ...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>皇家狗粮小型犬奶糕MIS30 1KG 哺乳孕期母犬 博美京巴幼犬通用型</td>
      <td>[皇家, 狗粮, 小型犬, 奶糕, MIS30,  , 1KG,  , 哺乳, 孕期, 母犬...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>皇家狗粮 拉布拉多成犬粮LR30 3KG*4 大型犬种 28省包邮</td>
      <td>[皇家, 狗粮,  , 拉布拉多, 成犬粮, LR30,  , 3KG, *, 4,  , ...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>皇家小型犬奶糕MIS30 3KG*2繁育期母离乳期幼犬博美狗粮28省包邮</td>
      <td>[皇家, 小型犬, 奶糕, MIS30,  , 3KG, *, 2, 繁育, 期母, 离乳期...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Royal Canin皇家 金毛幼犬粮AGR29/3.5KG*4 犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家,  , 金毛, 幼犬粮, AGR29, /, 3....</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Royal Canin皇家狗粮 柴犬成犬专用粮SIA26/3KG*2 犬主粮28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 成犬, 专用粮, SIA2...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>皇家狗粮 迷你雪纳瑞成犬粮SNZ25/3KG*2包 犬主粮 28省包邮</td>
      <td>[皇家, 狗粮,  , 迷你, 雪纳瑞, 成犬粮, SNZ25, /, 3KG, *, 2,...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>皇家小型幼犬狗粮MIJ31 0.8KG*3 博美狗茶杯腊肠犬 28省包邮</td>
      <td>[皇家, 小型, 幼犬, 狗粮, MIJ31,  , 0.8, KG, *, 3,  , 博...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Royal Canin皇家狗粮 拉布拉多幼犬粮ALR33 3KG*4 大型犬28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 拉布拉多, 幼犬粮, ALR33,...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Royal Canin皇家狗粮 柴犬幼犬专用粮SIJ29/3KG*2犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 幼犬, 专用粮, SIJ2...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>皇家小型老年犬狗粮SPR27 0.8KG*3  博美京巴腊肠犬粮 28省包邮</td>
      <td>[皇家, 小型, 老年犬, 狗粮, SPR27,  , 0.8, KG, *, 3,  , ...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>皇家狗粮 中型犬成犬粮M25 4KG*4 哈士奇沙皮通用型粮食 28省包邮</td>
      <td>[皇家, 狗粮,  , 中型犬, 成犬粮, M25,  , 4KG, *, 4,  , 哈士...</td>
    </tr>
    <tr>
      <th>55</th>
      <td>皇家狗粮 中型犬幼犬粮MEJ32 4KG*4萨摩耶哈士奇柯基狗粮28省包邮</td>
      <td>[皇家, 狗粮,  , 中型犬, 幼犬粮, MEJ32,  , 4KG, *, 4, 萨摩耶...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Royal Canin皇家狗粮 大型犬成犬粮GR26/15KG*2 犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 成犬粮, GR26, /...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Royal Canin皇家狗粮 西高地成犬粮WT21/3KG 专用粮 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 西高地, 成犬粮, WT21, /...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Royal Canin皇家狗粮 居家小型犬老年犬粮LIS24/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 老年犬, 粮, ...</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Royal Canin皇家狗粮 居家小型犬幼犬粮LIJ27/1.5KG*2 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 幼犬粮, LIJ...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Royal Canin皇家狗粮 拉布拉多成犬粮LR30 3KG大型犬狗粮热卖新品</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 拉布拉多, 成犬粮, LR30, ...</td>
    </tr>
    <tr>
      <th>72</th>
      <td>皇家狗粮 绝育呵护小型犬成犬粮MSA30 2KG 腊肠博美八哥犬全犬种</td>
      <td>[皇家, 狗粮,  , 绝育, 呵护, 小型犬, 成犬粮, MSA30,  , 2KG,  ...</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Royal Canin皇家狗粮 西高地成犬粮WT21/3KG*2 犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 西高地, 成犬粮, WT21, /...</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Royal Canin皇家狗粮 居家小型犬幼犬粮LIJ27/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 幼犬粮, LIJ...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Royal Canin皇家狗粮 居家小型犬成犬粮LIA21/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 成犬粮, LIA...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>皇家狗粮 大型犬繁育期/离乳期幼犬奶糕MAS30/4KG*2袋 28省包邮</td>
      <td>[皇家, 狗粮,  , 大型犬, 繁育, 期, /, 离乳期, 幼犬, 奶糕, MAS30,...</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Royal Canin皇家狗粮 约克夏幼犬粮APRY29/1.5KG*2犬主粮28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 约克夏, 幼犬粮, APRY29,...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Royal Canin皇家狗粮 中型犬奶糕MES30/10KG 离乳期幼犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 中型犬, 奶糕, MES30, /...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Royal Canin皇家狗粮 居家小型犬成犬粮LIA21/1.5KG*2  28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 成犬粮, LIA...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Royal Canin皇家狗粮 柴犬成犬专用粮SIA26/3KG 犬主粮狗粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 成犬, 专用粮, SIA2...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Royal Canin皇家狗粮 居家小型犬老年犬粮LIS24/1.5KG*2 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 老年犬, 粮, ...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>宝路狗粮狗主粮泰迪金毛拉布拉多大型小型通用型幼犬380g*10包</td>
      <td>[宝路, 狗粮, 狗主粮, 泰迪, 金毛, 拉布拉多, 大型, 小型, 通用型, 幼犬, 3...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>宝路狗粮狗主粮泰迪金毛拉布拉多比熊大型小型通用型幼犬380g*5包</td>
      <td>[宝路, 狗粮, 狗主粮, 泰迪, 金毛, 拉布拉多, 比熊, 大型, 小型, 通用型, 幼...</td>
    </tr>
    <tr>
      <th>128</th>
      <td>宝路狗粮狗主粮中型小型通用型成犬泰迪比熊牛肉味10斤500g*10包</td>
      <td>[宝路, 狗粮, 狗主粮, 中型, 小型, 通用型, 成犬, 泰迪, 比熊, 牛肉, 味, ...</td>
    </tr>
  </tbody>
</table>

</div>



```python
classi = []
for i in range(len(df_raw_1)):
    if not pd.isnull(df_raw_1.Classification[i]):
        classi.append(df_raw_1.Classification[i])
    elif df_raw_1.Breed[i] == '通用型':
        classi.append('犬粮')
    elif not pd.isnull(df_raw_1.Breed[i]):
        classi.append('专用粮')
    elif '专用粮' in  df_raw_1.item_name_cut[i]:
        classi.append('专用粮')
    elif '专用' in  df_raw_1.item_name_cut[i]:
        classi.append('专用粮')
    elif '奶糕' in  df_raw_1.item_name_cut[i]:
        classi.append('奶糕')
    else:
        classi.append('犬粮')
```

```python
df_raw_1.Classification = classi
```

```python
df_raw_1.Classification.value_counts(dropna=False)
```



```
犬粮     122
专用粮     51
奶糕      10
Name: Classification, dtype: int64
```



#### 适用品种

```python
df_raw_1.Breed.value_counts(dropna=False)
```



```
通用型       106
NaN        27
贵宾/泰迪      15
金毛          4
日本柴犬        4
斗牛犬         4
拉布拉多        4
约克夏梗        4
比熊          3
雪纳瑞         3
德国牧羊犬       3
吉娃娃         2
可卡          2
西部高地白梗      1
博美          1
Name: Breed, dtype: int64
```



存在27条空值

```python
df_raw_1[df_raw_1.Breed.isnull()][['item_name','item_name_cut']]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_name</th>
      <th>item_name_cut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Royal Canin皇家狗粮 小型犬老年犬狗粮SPR27/0.8kg公斤  犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 小型犬, 老年犬, 狗粮, SPR...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>皇家狗粮 小型犬成犬通用型PR27 2KG 博美迷你腊肠京巴八哥全犬种</td>
      <td>[皇家, 狗粮,  , 小型犬, 成犬, 通用型, PR27,  , 2KG,  , 博美,...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>皇家狗粮 小型犬成犬狗粮 通用型狗粮均衡营养减少牙石PR27/0.8KG</td>
      <td>[皇家, 狗粮,  , 小型犬, 成犬, 狗粮,  , 通用型, 狗粮, 均衡, 营养, 减...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>皇家狗粮小型犬8岁以上成犬粮SPR27 4KG*2  博美犬茶杯 28省包邮</td>
      <td>[皇家, 狗粮, 小型犬, 8, 岁, 以上, 成犬粮, SPR27,  , 4KG, *,...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>皇家狗粮 中型犬奶糕MES30 4KG*2繁育母犬离乳幼柯基萨摩28省包邮</td>
      <td>[皇家, 狗粮,  , 中型犬, 奶糕, MES30,  , 4KG, *, 2, 繁育, ...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Royal Canin皇家狗粮 大型犬成犬粮GR26/4KG*4包犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 成犬粮, GR26, /...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Royal Canin皇家狗粮 大型犬幼犬粮MAJ30/4KG*4袋  28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 幼犬粮, MAJ30, ...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Royal Canin皇家狗粮 雪纳瑞幼犬粮SNJ30/1.5KG*2袋 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 雪纳瑞, 幼犬粮, SNJ30, ...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>皇家小型犬奶糕MIS30 3KG*2繁育期母离乳期幼犬博美狗粮28省包邮</td>
      <td>[皇家, 小型犬, 奶糕, MIS30,  , 3KG, *, 2, 繁育, 期母, 离乳期...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>皇家小型幼犬狗粮MIJ31 0.8KG*3 博美狗茶杯腊肠犬 28省包邮</td>
      <td>[皇家, 小型, 幼犬, 狗粮, MIJ31,  , 0.8, KG, *, 3,  , 博...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>皇家小型老年犬狗粮SPR27 0.8KG*3  博美京巴腊肠犬粮 28省包邮</td>
      <td>[皇家, 小型, 老年犬, 狗粮, SPR27,  , 0.8, KG, *, 3,  , ...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>皇家狗粮 中型犬成犬粮M25 4KG*4 哈士奇沙皮通用型粮食 28省包邮</td>
      <td>[皇家, 狗粮,  , 中型犬, 成犬粮, M25,  , 4KG, *, 4,  , 哈士...</td>
    </tr>
    <tr>
      <th>55</th>
      <td>皇家狗粮 中型犬幼犬粮MEJ32 4KG*4萨摩耶哈士奇柯基狗粮28省包邮</td>
      <td>[皇家, 狗粮,  , 中型犬, 幼犬粮, MEJ32,  , 4KG, *, 4, 萨摩耶...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Royal Canin皇家狗粮 大型犬成犬粮GR26/15KG*2 犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 成犬粮, GR26, /...</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Royal Canin皇家狗粮 德牧成犬粮GS24/12KG*2犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 德牧, 成犬粮, GS24, /,...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Royal Canin皇家狗粮 居家小型犬老年犬粮LIS24/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 老年犬, 粮, ...</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Royal Canin皇家狗粮 居家小型犬幼犬粮LIJ27/1.5KG*2 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 幼犬粮, LIJ...</td>
    </tr>
    <tr>
      <th>72</th>
      <td>皇家狗粮 绝育呵护小型犬成犬粮MSA30 2KG 腊肠博美八哥犬全犬种</td>
      <td>[皇家, 狗粮,  , 绝育, 呵护, 小型犬, 成犬粮, MSA30,  , 2KG,  ...</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Royal Canin皇家狗粮 西高地成犬粮WT21/3KG*2 犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 西高地, 成犬粮, WT21, /...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Royal Canin皇家狗粮 居家小型犬成犬粮LIA21/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 成犬粮, LIA...</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Royal Canin皇家狗粮 雪纳瑞幼犬粮SNJ30/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 雪纳瑞, 幼犬粮, SNJ30, ...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>皇家狗粮 大型犬繁育期/离乳期幼犬奶糕MAS30/4KG*2袋 28省包邮</td>
      <td>[皇家, 狗粮,  , 大型犬, 繁育, 期, /, 离乳期, 幼犬, 奶糕, MAS30,...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Royal Canin皇家狗粮 居家小型犬成犬粮LIA21/1.5KG*2  28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 成犬粮, LIA...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Royal Canin皇家狗粮 居家小型犬老年犬粮LIS24/1.5KG*2 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 老年犬, 粮, ...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>宝路狗粮狗主粮泰迪金毛拉布拉多比熊大型小型通用型幼犬380g*5包</td>
      <td>[宝路, 狗粮, 狗主粮, 泰迪, 金毛, 拉布拉多, 比熊, 大型, 小型, 通用型, 幼...</td>
    </tr>
    <tr>
      <th>120</th>
      <td>宝路狗粮狗主粮泰迪比熊贵宾中型小型通用型成犬鸡肉味15斤7.5kg</td>
      <td>[宝路, 狗粮, 狗主粮, 泰迪, 比熊, 贵宾, 中型, 小型, 通用型, 成犬, 鸡肉,...</td>
    </tr>
    <tr>
      <th>128</th>
      <td>宝路狗粮狗主粮中型小型通用型成犬泰迪比熊牛肉味10斤500g*10包</td>
      <td>[宝路, 狗粮, 狗主粮, 中型, 小型, 通用型, 成犬, 泰迪, 比熊, 牛肉, 味, ...</td>
    </tr>
  </tbody>
</table>

</div>



```python
bre = []
for i in range(len(df_raw_1)):
    if not pd.isnull(df_raw_1.Breed[i]):
        bre.append(df_raw_1.Breed[i])
    elif '雪纳瑞' in  df_raw_1.item_name_cut[i]:
        bre.append('雪纳瑞')
    elif  '德牧' in  df_raw_1.item_name_cut[i]:
        bre.append('德国牧羊犬')
    elif  '西高地' in  df_raw_1.item_name_cut[i]:
        bre.append('西部高地白梗')
    elif  '博美' in  df_raw_1.item_name_cut[i] and df_raw_1.Classification[i] =='奶糕':
        bre.append('博美')
    else:
#         bre.append(df_raw_1.item_name_cut[i])
        bre.append('通用型')
```

```python
bre
```



```
['德国牧羊犬',
 '日本柴犬',
 '通用型',
 '雪纳瑞',
 '拉布拉多',
 '通用型',
 '通用型',
 '约克夏梗',
 '通用型',
 '斗牛犬',
 '通用型',
 '通用型',
 '吉娃娃',
 '通用型',
 '通用型',
 '贵宾/泰迪',
 '通用型',
 '贵宾/泰迪',
 '贵宾/泰迪',
 '通用型',
 '贵宾/泰迪',
 '通用型',
 '比熊',
 '通用型',
 '贵宾/泰迪',
 '通用型',
 '金毛',
 '通用型',
 '通用型',
 '约克夏梗',
 '通用型',
 '贵宾/泰迪',
 '通用型',
 '拉布拉多',
 '贵宾/泰迪',
 '雪纳瑞',
 '博美',
 '金毛',
 '通用型',
 '通用型',
 '金毛',
 '日本柴犬',
 '通用型',
 '雪纳瑞',
 '通用型',
 '拉布拉多',
 '日本柴犬',
 '通用型',
 '斗牛犬',
 '通用型',
 '通用型',
 '比熊',
 '通用型',
 '贵宾/泰迪',
 '斗牛犬',
 '通用型',
 '贵宾/泰迪',
 '通用型',
 '通用型',
 '贵宾/泰迪',
 '通用型',
 '可卡',
 '德国牧羊犬',
 '约克夏梗',
 '吉娃娃',
 '德国牧羊犬',
 '西部高地白梗',
 '通用型',
 '通用型',
 '可卡',
 '拉布拉多',
 '通用型',
 '通用型',
 '西部高地白梗',
 '通用型',
 '通用型',
 '通用型',
 '斗牛犬',
 '金毛',
 '通用型',
 '雪纳瑞',
 '通用型',
 '德国牧羊犬',
 '通用型',
 '约克夏梗',
 '通用型',
 '通用型',
 '日本柴犬',
 '通用型',
 '通用型',
 '贵宾/泰迪',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '贵宾/泰迪',
 '通用型',
 '博美',
 '雪纳瑞',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '比熊',
 '通用型',
 '贵宾/泰迪',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '贵宾/泰迪',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '通用型',
 '贵宾/泰迪']
```



```python
for i in bre:
    if type(i) is list:
        print(i)
```

```python
df_raw_1.Breed = bre
```

```python
df_raw_1.Breed.value_counts(dropna=False)
```



```
通用型       128
贵宾/泰迪      15
雪纳瑞         5
金毛          4
德国牧羊犬       4
日本柴犬        4
斗牛犬         4
拉布拉多        4
约克夏梗        4
比熊          3
西部高地白梗      2
博美          2
吉娃娃         2
可卡          2
Name: Breed, dtype: int64
```



#### 适用体型

```python
df_raw_1.BodyType.value_counts(dropna=False)
```



```
通用型     47
小型犬     46
NaN     46
大型犬     20
中小型犬    16
中型犬      6
中大型犬     2
Name: BodyType, dtype: int64
```



存在46条空值

```python
df_raw_1[df_raw_1.BodyType.isnull()][['item_name','item_name_cut']]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_name</th>
      <th>item_name_cut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Royal Canin皇家狗粮 柴犬幼犬专用粮SIJ29/3KG 犬主粮狗粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 幼犬, 专用粮, SIJ2...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Royal Canin皇家狗粮 迷你雪纳瑞成犬粮SNZ25/3KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 迷你, 雪纳瑞, 成犬粮, SNZ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Royal Canin皇家狗粮 拉布拉多幼犬粮ALR33 3KG 大型犬全犬种热卖</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 拉布拉多, 幼犬粮, ALR33,...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Royal Canin皇家狗粮 法国斗牛犬幼犬粮FBJ30/3KG 法斗粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 法国, 斗牛犬, 幼犬粮, FBJ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>皇家狗粮 吉娃娃成犬专用粮食C28 1.5KG*2  小型犬 28省包邮</td>
      <td>[皇家, 狗粮,  , 吉娃娃, 成犬, 专用, 粮食, C28,  , 1.5, KG, ...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>皇家官方旗舰店 贵宾泰迪狗粮8岁以上老年犬PDA26 3KG热卖新品</td>
      <td>[皇家, 官方, 旗舰店,  , 贵宾, 泰迪, 狗粮, 8, 岁, 以上, 老年犬, PD...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>皇家狗粮 拉布拉多成犬粮LR30 3KG*4 大型犬种 28省包邮</td>
      <td>[皇家, 狗粮,  , 拉布拉多, 成犬粮, LR30,  , 3KG, *, 4,  , ...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>皇家8岁+贵宾泰迪老年犬粮PDA26/3KG*2 犬主粮贵宾狗粮 28省包邮</td>
      <td>[皇家, 8, 岁, +, 贵宾, 泰迪, 老年犬, 粮, PDA26, /, 3KG, *...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Royal Canin皇家狗粮 雪纳瑞幼犬粮SNJ30/1.5KG*2袋 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 雪纳瑞, 幼犬粮, SNJ30, ...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>皇家小型犬奶糕MIS30 3KG*2繁育期母离乳期幼犬博美狗粮28省包邮</td>
      <td>[皇家, 小型犬, 奶糕, MIS30,  , 3KG, *, 2, 繁育, 期母, 离乳期...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Royal Canin皇家狗粮 金毛成犬粮GR25/3.5KG*4 犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 金毛, 成犬粮, GR25, /,...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Royal Canin皇家狗粮 柴犬成犬专用粮SIA26/3KG*2 犬主粮28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 成犬, 专用粮, SIA2...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>皇家狗粮 迷你雪纳瑞成犬粮SNZ25/3KG*2包 犬主粮 28省包邮</td>
      <td>[皇家, 狗粮,  , 迷你, 雪纳瑞, 成犬粮, SNZ25, /, 3KG, *, 2,...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>皇家小型幼犬狗粮MIJ31 0.8KG*3 博美狗茶杯腊肠犬 28省包邮</td>
      <td>[皇家, 小型, 幼犬, 狗粮, MIJ31,  , 0.8, KG, *, 3,  , 博...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Royal Canin皇家狗粮 拉布拉多幼犬粮ALR33 3KG*4 大型犬28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 拉布拉多, 幼犬粮, ALR33,...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Royal Canin皇家狗粮 柴犬幼犬专用粮SIJ29/3KG*2犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 幼犬, 专用粮, SIJ2...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>皇家狗粮 法国斗牛犬成犬粮FBA26/3KG*2 犬主粮 法斗粮 28省包邮</td>
      <td>[皇家, 狗粮,  , 法国, 斗牛犬, 成犬粮, FBA26, /, 3KG, *, 2,...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>皇家小型老年犬狗粮SPR27 0.8KG*3  博美京巴腊肠犬粮 28省包邮</td>
      <td>[皇家, 小型, 老年犬, 狗粮, SPR27,  , 0.8, KG, *, 3,  , ...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>皇家狗粮 中型犬成犬粮M25 4KG*4 哈士奇沙皮通用型粮食 28省包邮</td>
      <td>[皇家, 狗粮,  , 中型犬, 成犬粮, M25,  , 4KG, *, 4,  , 哈士...</td>
    </tr>
    <tr>
      <th>53</th>
      <td>RC皇家贵宾成犬粮 PD30/3KG*2 泰迪狗粮 犬主粮 28省包邮</td>
      <td>[RC, 皇家, 贵宾, 成犬粮,  , PD30, /, 3KG, *, 2,  , 泰迪...</td>
    </tr>
    <tr>
      <th>54</th>
      <td>皇家狗粮 法国斗牛犬幼犬粮FBJ30/3KG*2 法斗粮 犬主粮 28省包邮</td>
      <td>[皇家, 狗粮,  , 法国, 斗牛犬, 幼犬粮, FBJ30, /, 3KG, *, 2,...</td>
    </tr>
    <tr>
      <th>55</th>
      <td>皇家狗粮 中型犬幼犬粮MEJ32 4KG*4萨摩耶哈士奇柯基狗粮28省包邮</td>
      <td>[皇家, 狗粮,  , 中型犬, 幼犬粮, MEJ32,  , 4KG, *, 4, 萨摩耶...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Royal Canin皇家狗粮 大型犬成犬粮GR26/15KG*2 犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 成犬粮, GR26, /...</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Royal Canin皇家狗粮 约克夏幼犬粮APRY29/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 约克夏, 幼犬粮, APRY29,...</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Royal Canin皇家狗粮 德牧幼犬粮AGS30/12KG*2 犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 德牧, 幼犬粮, AGS30, /...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Royal Canin皇家狗粮 西高地成犬粮WT21/3KG 专用粮 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 西高地, 成犬粮, WT21, /...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Royal Canin皇家狗粮 居家小型犬老年犬粮LIS24/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 老年犬, 粮, ...</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Royal Canin皇家狗粮 居家小型犬幼犬粮LIJ27/1.5KG*2 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 幼犬粮, LIJ...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Royal Canin皇家狗粮 拉布拉多成犬粮LR30 3KG大型犬狗粮热卖新品</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 拉布拉多, 成犬粮, LR30, ...</td>
    </tr>
    <tr>
      <th>72</th>
      <td>皇家狗粮 绝育呵护小型犬成犬粮MSA30 2KG 腊肠博美八哥犬全犬种</td>
      <td>[皇家, 狗粮,  , 绝育, 呵护, 小型犬, 成犬粮, MSA30,  , 2KG,  ...</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Royal Canin皇家狗粮 西高地成犬粮WT21/3KG*2 犬主粮 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 西高地, 成犬粮, WT21, /...</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Royal Canin皇家狗粮 法国斗牛犬成犬粮FBA26/3KG 法斗粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 法国, 斗牛犬, 成犬粮, FBA...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Royal Canin皇家狗粮 居家小型犬成犬粮LIA21/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 成犬粮, LIA...</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Royal Canin皇家狗粮 雪纳瑞幼犬粮SNJ30/1.5KG 犬主粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 雪纳瑞, 幼犬粮, SNJ30, ...</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Royal Canin皇家狗粮 德牧成犬粮GS24 12KG 大型犬28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 德牧, 成犬粮, GS24,  ,...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>皇家狗粮 大型犬繁育期/离乳期幼犬奶糕MAS30/4KG*2袋 28省包邮</td>
      <td>[皇家, 狗粮,  , 大型犬, 繁育, 期, /, 离乳期, 幼犬, 奶糕, MAS30,...</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Royal Canin皇家狗粮 约克夏幼犬粮APRY29/1.5KG*2犬主粮28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 约克夏, 幼犬粮, APRY29,...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Royal Canin皇家狗粮 居家小型犬成犬粮LIA21/1.5KG*2  28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 成犬粮, LIA...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Royal Canin皇家狗粮 柴犬成犬专用粮SIA26/3KG 犬主粮狗粮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 成犬, 专用粮, SIA2...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Royal Canin皇家狗粮 居家小型犬老年犬粮LIS24/1.5KG*2 28省包邮</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 居家, 小型犬, 老年犬, 粮, ...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>宝路狗粮狗主粮泰迪金毛拉布拉多大型小型通用型幼犬380g*10包</td>
      <td>[宝路, 狗粮, 狗主粮, 泰迪, 金毛, 拉布拉多, 大型, 小型, 通用型, 幼犬, 3...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>宝路狗粮狗主粮泰迪金毛拉布拉多比熊大型小型通用型幼犬380g*5包</td>
      <td>[宝路, 狗粮, 狗主粮, 泰迪, 金毛, 拉布拉多, 比熊, 大型, 小型, 通用型, 幼...</td>
    </tr>
    <tr>
      <th>120</th>
      <td>宝路狗粮狗主粮泰迪比熊贵宾中型小型通用型成犬鸡肉味15斤7.5kg</td>
      <td>[宝路, 狗粮, 狗主粮, 泰迪, 比熊, 贵宾, 中型, 小型, 通用型, 成犬, 鸡肉,...</td>
    </tr>
    <tr>
      <th>128</th>
      <td>宝路狗粮狗主粮中型小型通用型成犬泰迪比熊牛肉味10斤500g*10包</td>
      <td>[宝路, 狗粮, 狗主粮, 中型, 小型, 通用型, 成犬, 泰迪, 比熊, 牛肉, 味, ...</td>
    </tr>
    <tr>
      <th>143</th>
      <td>比瑞吉新鲜全期犬粮2kg 比瑞吉新鲜粮 天然粮 通用狗粮500g*4小包</td>
      <td>[比瑞吉, 新鲜, 全期犬粮, 2kg,  , 比瑞吉, 新鲜, 粮,  , 天然粮,  ,...</td>
    </tr>
    <tr>
      <th>179</th>
      <td>海洋之星狗粮试吃装泰迪比熊玩具犬幼犬成犬全犬期小颗粒三包*30g</td>
      <td>[海洋之星, 狗粮, 试吃装, 泰迪, 比熊, 玩具犬, 幼犬, 成犬, 全犬期, 小颗粒,...</td>
    </tr>
  </tbody>
</table>

</div>



```python
bodytp = []
for i in range(len(df_raw_1)):
    if not pd.isnull(df_raw_1.BodyType[i]):
        bodytp.append(df_raw_1.BodyType[i])
    elif '通用型' in  df_raw_1.item_name_cut[i]:
        bodytp.append('通用型')
    elif '通用' in  df_raw_1.item_name_cut[i]:
        bodytp.append('通用型')
    elif '小型犬' in  df_raw_1.item_name_cut[i]:
         bodytp.append('小型犬')
    elif '小型' in  df_raw_1.item_name_cut[i]:
         bodytp.append('小型犬')
    elif '中型犬' in  df_raw_1.item_name_cut[i]:
         bodytp.append('中型犬')
    elif '中型' in  df_raw_1.item_name_cut[i]:
         bodytp.append('中型犬')
    elif '大型犬' in  df_raw_1.item_name_cut[i]:
         bodytp.append('大型犬')
    elif '大型' in  df_raw_1.item_name_cut[i]:
         bodytp.append('大型犬')
    else:
#         bodytp.append(df_raw_1.item_name_cut[i])
        bodytp.append('通用型')
```

```python
df_raw_1.BodyType = bodytp
```

```python
df_raw_1.BodyType.value_counts(dropna=False)
```



```
通用型     75
小型犬     56
大型犬     27
中小型犬    16
中型犬      7
中大型犬     2
Name: BodyType, dtype: int64
```



```python
df_raw_1.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_id</th>
      <th>item_name</th>
      <th>TradeName</th>
      <th>price</th>
      <th>total_sale</th>
      <th>month_sale</th>
      <th>accum_comm</th>
      <th>TM_points</th>
      <th>CollectCount</th>
      <th>Tastes</th>
      <th>...</th>
      <th>ApplicablePhase</th>
      <th>Brand</th>
      <th>Classification</th>
      <th>Breed</th>
      <th>Manufacturer</th>
      <th>Weight</th>
      <th>Origin</th>
      <th>ManufacturerAddress</th>
      <th>RecipeTastePrescription</th>
      <th>item_name_cut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19045209534</td>
      <td>Royal Canin皇家狗粮 德牧幼犬粮AGS30 12KG 大型犬狗粮28省包邮</td>
      <td>德国牧羊犬幼犬专用粮 12kg</td>
      <td>650.0</td>
      <td>2046</td>
      <td>39</td>
      <td>418</td>
      <td>325</td>
      <td>348</td>
      <td>其他</td>
      <td>...</td>
      <td>幼犬</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>德国牧羊犬</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>上海</td>
      <td>NaN</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 德牧, 幼犬粮, AGS30,  ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>547144906363</td>
      <td>Royal Canin皇家狗粮 柴犬幼犬专用粮SIJ29/3KG 犬主粮狗粮</td>
      <td>日本柴犬幼犬 3000g</td>
      <td>285.0</td>
      <td>560</td>
      <td>39</td>
      <td>132</td>
      <td>142</td>
      <td>200</td>
      <td>其他</td>
      <td>...</td>
      <td>幼犬</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>日本柴犬</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>NaN</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 幼犬, 专用粮, SIJ2...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26255560615</td>
      <td>皇家狗粮 大型犬奶糕MAS30 4KG哺乳孕期犬离乳期幼犬牧羊阿拉斯加</td>
      <td>(大型犬)离乳期奶糕 4kg</td>
      <td>301.0</td>
      <td>1571</td>
      <td>27</td>
      <td>390</td>
      <td>150</td>
      <td>554</td>
      <td>其他</td>
      <td>...</td>
      <td>离乳期</td>
      <td>ROYAL CANIN/皇家</td>
      <td>奶糕</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>4000</td>
      <td>中国</td>
      <td>上海</td>
      <td>NaN</td>
      <td>[皇家, 狗粮,  , 大型犬, 奶糕, MAS30,  , 4KG, 哺乳, 孕期, 犬离...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39792870853</td>
      <td>Royal Canin皇家狗粮 迷你雪纳瑞成犬粮SNZ25/3KG 犬主粮</td>
      <td>雪纳瑞成犬 3000g</td>
      <td>260.0</td>
      <td>2113</td>
      <td>32</td>
      <td>281</td>
      <td>130</td>
      <td>379</td>
      <td>其他</td>
      <td>...</td>
      <td>成犬</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>雪纳瑞</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海奉贤区肖南路475号</td>
      <td>NaN</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 迷你, 雪纳瑞, 成犬粮, SNZ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>547165690913</td>
      <td>Royal Canin皇家狗粮 拉布拉多幼犬粮ALR33 3KG 大型犬全犬种热卖</td>
      <td>拉布拉多幼犬 3000g</td>
      <td>210.0</td>
      <td>643</td>
      <td>48</td>
      <td>173</td>
      <td>105</td>
      <td>318</td>
      <td>其他</td>
      <td>...</td>
      <td>幼犬</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>拉布拉多</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>NaN</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 拉布拉多, 幼犬粮, ALR33,...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>

</div>



#### 口味配方

```python
df_raw_1.RecipeTastePrescription.value_counts(dropna=False)
```



```
NaN                             104
无                                 6
牛肉口味                              5
鸡肉味                               5
海藻胡萝卜粒配方                          4
牛肉、肝、蔬菜及谷物味                       3
牛肉味                               3
无谷六种肉配方                           2
无谷物深海鱼大颗粒配方                       2
肉果蔬粗粮配方                           2
三文鱼                               2
鱼肉味                               2
深海鱼配方                             2
牛肉口味、蔬菜及谷物配方                      2
鸡肉口味                              2
深海鱼味                              2
牛肉、鸡肉、蔬菜及谷物配方                     2
精选肉类、奶、蔬菜及谷物味                     2
牛肉、肝、蔬菜及谷物配方                      1
鱼肉                                1
弱体质调理                             1
泌尿道调理                             1
牛奶夹心酥                             1
成犬无谷物深海鱼小颗粒1.5kg                  1
牛肉鸡肉蔬菜味                           1
精选海洋营养加强三文鱼成犬配方                   1
无谷物三文鱼玩具犬配方                       1
Superior（优级）三文鱼控制体重配方大颗粒12kg      1
无谷物深海鱼幼犬配方小颗粒                     1
绝育低卡调理                            1
肠道调理                              1
无谷物深海鱼幼犬配方大颗粒                     1
精选肉类、奶、蔬菜及谷物配方                    1
幼犬无谷物深海鱼小颗粒6kg                    1
皮肤调理                              1
精选海洋营养加强三文鱼幼犬配方                   1
牛奶、蔬菜及谷物配方                        1
深海鱼、肝、蔬菜及谷物味                      1
无谷物三文鱼大颗粒配方                       1
鸡肉、肝、蔬菜及谷物味                       1
Superior（优级）三文鱼控制体重小颗粒6kg         1
Superior（优级）体重控制小颗粒1.5kg          1
心脏调理                              1
鱼味                                1
精选海洋营养加强体重控制三文鱼配方                 1
肾脏调理                              1
精选肉类、肝、蔬菜及谷物味                     1
鸡肉、肝、蔬菜及谷物配方                      1
无谷物深海鱼小颗粒配方                       1
Name: RecipeTastePrescription, dtype: int64
```



有104条空值

```python
rectaste = []
for i in range(len(df_raw_1)):
    if df_raw_1.Brand[i] == 'ROYAL CANIN/皇家':
        rectaste.append('无')
    elif df_raw_1.Manufacturer[i] == '玛氏食品（中国）有限公司':
        if not pd.isnull(df_raw_1.RecipeTastePrescription[i]):
            rectaste.append(df_raw_1.RecipeTastePrescription[i])
        elif df_raw_1.Tastes[i]!='其他':
            rectaste.append(df_raw_1.Tastes[i])
        else:
            rectaste.append('无')   
    elif df_raw_1.Brand[i] == 'Nature Bridge/比瑞吉':
        if not pd.isnull(df_raw_1.RecipeTastePrescription[i]):
            rectaste.append(df_raw_1.RecipeTastePrescription[i])
        else:
            rectaste.append('无')
    elif df_raw_1.Brand[i] == 'Fish4Dogs/海洋之星':
        if not pd.isnull(df_raw_1.RecipeTastePrescription[i]):
            rectaste.append(df_raw_1.RecipeTastePrescription[i])
        else:
            rectaste.append('鱼肉味')
```

```python
df_raw_1.RecipeTastePrescription = rectaste
```

```python
df_raw_1.RecipeTastePrescription.value_counts(dropna=False)
```



```
无                               106
鸡肉味                               6
牛肉味                               5
牛肉口味                              5
海藻胡萝卜粒配方                          4
鱼肉味                               3
牛肉、肝、蔬菜及谷物味                       3
肉果蔬粗粮配方                           2
无谷六种肉配方                           2
深海鱼配方                             2
无谷物深海鱼大颗粒配方                       2
三文鱼                               2
牛肉口味、蔬菜及谷物配方                      2
鸡肉口味                              2
深海鱼味                              2
牛肉、鸡肉、蔬菜及谷物配方                     2
精选肉类、奶、蔬菜及谷物味                     2
牛肉鸡肉蔬菜味                           1
鱼肉                                1
精选海洋营养加强三文鱼成犬配方                   1
无谷物深海鱼幼犬配方小颗粒                     1
肠道调理                              1
成犬无谷物深海鱼小颗粒1.5kg                  1
无谷物深海鱼幼犬配方大颗粒                     1
绝育低卡调理                            1
牛肉、肝、蔬菜及谷物配方                      1
泌尿道调理                             1
牛奶夹心酥                             1
无谷物三文鱼玩具犬配方                       1
弱体质调理                             1
无谷物深海鱼小颗粒配方                       1
精选肉类、奶、蔬菜及谷物配方                    1
鸡肉、肝、蔬菜及谷物味                       1
幼犬无谷物深海鱼小颗粒6kg                    1
皮肤调理                              1
精选海洋营养加强三文鱼幼犬配方                   1
牛奶、蔬菜及谷物配方                        1
深海鱼、肝、蔬菜及谷物味                      1
无谷物三文鱼大颗粒配方                       1
Superior（优级）三文鱼控制体重小颗粒6kg         1
鸡肉、肝、蔬菜及谷物配方                      1
Superior（优级）体重控制小颗粒1.5kg          1
心脏调理                              1
鱼味                                1
精选海洋营养加强体重控制三文鱼配方                 1
肾脏调理                              1
精选肉类、肝、蔬菜及谷物味                     1
Superior（优级）三文鱼控制体重配方大颗粒12kg      1
Name: RecipeTastePrescription, dtype: int64
```



数据清理、整合
--------------

### 口味配方

```python
df_raw_1[df_raw_1.Brand == 'ROYAL CANIN/皇家']['RecipeTastePrescription'].value_counts()
```



```
无    91
Name: RecipeTastePrescription, dtype: int64
```



```python
df_raw_1[df_raw_1.Manufacturer == '玛氏食品（中国）有限公司']['RecipeTastePrescription'].value_counts()
```



```
鸡肉味               6
牛肉口味              5
牛肉味               5
牛肉、肝、蔬菜及谷物味       3
深海鱼味              2
精选肉类、奶、蔬菜及谷物味     2
无                 2
牛肉、鸡肉、蔬菜及谷物配方     2
牛肉口味、蔬菜及谷物配方      2
鸡肉口味              2
牛肉、肝、蔬菜及谷物配方      1
鸡肉、肝、蔬菜及谷物配方      1
牛奶夹心酥             1
精选肉类、肝、蔬菜及谷物味     1
牛肉鸡肉蔬菜味           1
精选肉类、奶、蔬菜及谷物配方    1
牛奶、蔬菜及谷物配方        1
鸡肉、肝、蔬菜及谷物味       1
深海鱼、肝、蔬菜及谷物味      1
Name: RecipeTastePrescription, dtype: int64
```



```python
df_raw_1[df_raw_1.Brand == 'Nature Bridge/比瑞吉']['RecipeTastePrescription'].value_counts()
```



```
无           13
海藻胡萝卜粒配方     4
肉果蔬粗粮配方      2
无谷六种肉配方      2
弱体质调理        1
绝育低卡调理       1
肠道调理         1
泌尿道调理        1
心脏调理         1
皮肤调理         1
肾脏调理         1
Name: RecipeTastePrescription, dtype: int64
```



```python
df_raw_1[df_raw_1.Brand == 'Fish4Dogs/海洋之星']['RecipeTastePrescription'].value_counts()
```



```
鱼肉味                             3
深海鱼配方                           2
无谷物深海鱼大颗粒配方                     2
三文鱼                             2
Superior（优级）体重控制小颗粒1.5kg        1
无谷物三文鱼玩具犬配方                     1
无谷物深海鱼幼犬配方大颗粒                   1
鱼味                              1
无谷物深海鱼幼犬配方小颗粒                   1
Superior（优级）三文鱼控制体重小颗粒6kg       1
无谷物深海鱼小颗粒配方                     1
成犬无谷物深海鱼小颗粒1.5kg                1
无谷物三文鱼大颗粒配方                     1
精选海洋营养加强三文鱼幼犬配方                 1
精选海洋营养加强体重控制三文鱼配方               1
鱼肉                              1
幼犬无谷物深海鱼小颗粒6kg                  1
精选海洋营养加强三文鱼成犬配方                 1
Superior（优级）三文鱼控制体重配方大颗粒12kg    1
Name: RecipeTastePrescription, dtype: int64
```



```python
b=jieba.lcut(df_raw_1.RecipeTastePrescription[161])
b
```



```
['Superior', '（', '优级', '）', '三文鱼', '控制', '体重', '配方', '大颗粒', '12kg']
```



```python
rectaste_2 = []
for i in range(len(df_raw_1)):
    cut_i=jieba.lcut(df_raw_1.RecipeTastePrescription[i])
    if df_raw_1.Brand[i] == 'ROYAL CANIN/皇家':
        rectaste_2.append(df_raw_1.RecipeTastePrescription[i])
    elif df_raw_1.Manufacturer[i] == '玛氏食品（中国）有限公司':
        if '蔬菜' in  cut_i:
            rectaste_2.append('营养均衡配方')
        elif '牛肉'in cut_i:
            rectaste_2.append('牛肉口味配方')
        elif '鸡肉'in cut_i:
            rectaste_2.append('鸡肉口味配方')  
        elif '深海鱼'in cut_i:
            rectaste_2.append('深海鱼味配方')      
        else:
            rectaste_2.append(df_raw_1.RecipeTastePrescription[i])   
    elif df_raw_1.Brand[i] == 'Nature Bridge/比瑞吉':
        if '六种' in  cut_i:
            rectaste_2.append('无谷多肉配方')
        elif  '果蔬' in  cut_i or '胡萝卜' in cut_i:
            rectaste_2.append('营养均衡配方')
        else:rectaste_2.append(df_raw_1.RecipeTastePrescription[i]) 
    elif df_raw_1.Brand[i] == 'Fish4Dogs/海洋之星':
        if '无谷物' in  cut_i and '三文鱼' in cut_i:
            rectaste_2.append('无谷三文鱼配方')
        elif '无谷物' in  cut_i and '深海鱼' in cut_i:
            rectaste_2.append('无谷深海鱼配方') 
        elif  '深海鱼' in cut_i :
            rectaste_2.append('深海鱼味配方') 
        elif  '三文鱼' in cut_i:
            rectaste_2.append('三文鱼味配方') 
        else:
            rectaste_2.append('鱼肉口味配方')  

        
```

```python
df_raw_1['RecipeTastePrescription_v2'] = rectaste_2
```

```python
pd.Series(df_raw_1.RecipeTastePrescription_v2).value_counts()
```



```
无          106
营养均衡配方      23
牛肉口味配方      10
鸡肉口味配方       8
三文鱼味配方       7
无谷深海鱼配方      7
鱼肉口味配方       6
深海鱼味配方       4
无谷多肉配方       2
无谷三文鱼配方      2
皮肤调理         1
弱体质调理        1
绝育低卡调理       1
肠道调理         1
泌尿道调理        1
牛奶夹心酥        1
心脏调理         1
肾脏调理         1
Name: RecipeTastePrescription_v2, dtype: int64
```



```python
df_raw_1
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_id</th>
      <th>item_name</th>
      <th>TradeName</th>
      <th>price</th>
      <th>total_sale</th>
      <th>month_sale</th>
      <th>accum_comm</th>
      <th>TM_points</th>
      <th>CollectCount</th>
      <th>Tastes</th>
      <th>...</th>
      <th>Brand</th>
      <th>Classification</th>
      <th>Breed</th>
      <th>Manufacturer</th>
      <th>Weight</th>
      <th>Origin</th>
      <th>ManufacturerAddress</th>
      <th>RecipeTastePrescription</th>
      <th>item_name_cut</th>
      <th>RecipeTastePrescription_v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19045209534</td>
      <td>Royal Canin皇家狗粮 德牧幼犬粮AGS30 12KG 大型犬狗粮28省包邮</td>
      <td>德国牧羊犬幼犬专用粮 12kg</td>
      <td>650.0</td>
      <td>2046</td>
      <td>39</td>
      <td>418</td>
      <td>325</td>
      <td>348</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>德国牧羊犬</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 德牧, 幼犬粮, AGS30,  ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>1</th>
      <td>547144906363</td>
      <td>Royal Canin皇家狗粮 柴犬幼犬专用粮SIJ29/3KG 犬主粮狗粮</td>
      <td>日本柴犬幼犬 3000g</td>
      <td>285.0</td>
      <td>560</td>
      <td>39</td>
      <td>132</td>
      <td>142</td>
      <td>200</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>日本柴犬</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 幼犬, 专用粮, SIJ2...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26255560615</td>
      <td>皇家狗粮 大型犬奶糕MAS30 4KG哺乳孕期犬离乳期幼犬牧羊阿拉斯加</td>
      <td>(大型犬)离乳期奶糕 4kg</td>
      <td>301.0</td>
      <td>1571</td>
      <td>27</td>
      <td>390</td>
      <td>150</td>
      <td>554</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>奶糕</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>4000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮,  , 大型犬, 奶糕, MAS30,  , 4KG, 哺乳, 孕期, 犬离...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39792870853</td>
      <td>Royal Canin皇家狗粮 迷你雪纳瑞成犬粮SNZ25/3KG 犬主粮</td>
      <td>雪纳瑞成犬 3000g</td>
      <td>260.0</td>
      <td>2113</td>
      <td>32</td>
      <td>281</td>
      <td>130</td>
      <td>379</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>雪纳瑞</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海奉贤区肖南路475号</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 迷你, 雪纳瑞, 成犬粮, SNZ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>4</th>
      <td>547165690913</td>
      <td>Royal Canin皇家狗粮 拉布拉多幼犬粮ALR33 3KG 大型犬全犬种热卖</td>
      <td>拉布拉多幼犬 3000g</td>
      <td>210.0</td>
      <td>643</td>
      <td>48</td>
      <td>173</td>
      <td>105</td>
      <td>318</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>拉布拉多</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 拉布拉多, 幼犬粮, ALR33,...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>5</th>
      <td>19086681580</td>
      <td>Royal Canin皇家狗粮 小型犬老年犬狗粮SPR27/0.8kg公斤  犬主粮</td>
      <td>(小型犬)老年犬犬粮 800G</td>
      <td>66.0</td>
      <td>1906</td>
      <td>46</td>
      <td>473</td>
      <td>33</td>
      <td>205</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>800</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 小型犬, 老年犬, 狗粮, SPR...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>6</th>
      <td>18965425186</td>
      <td>Royal Canin皇家狗粮 大型犬幼犬粮MAJ30/15KG  犬主粮28省包邮</td>
      <td>(大型犬)幼犬犬粮 15KG</td>
      <td>623.0</td>
      <td>5018</td>
      <td>56</td>
      <td>771</td>
      <td>311</td>
      <td>1163</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>15000</td>
      <td>中国</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 幼犬粮, MAJ30, ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>7</th>
      <td>26217124504</td>
      <td>Royal Canin皇家狗粮 约克夏成犬专用粮PRY28/1.5KG 犬主粮</td>
      <td>约克夏梗(小型犬)成犬专用粮 1.5KG</td>
      <td>135.0</td>
      <td>2057</td>
      <td>31</td>
      <td>329</td>
      <td>67</td>
      <td>238</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>约克夏梗</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>1500</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 约克夏, 成犬, 专用粮, PRY...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>8</th>
      <td>19738658492</td>
      <td>皇家狗粮 小型犬成犬通用型PR27 2KG 博美迷你腊肠京巴八哥全犬种</td>
      <td>(小型犬)成犬犬粮 2000g</td>
      <td>141.0</td>
      <td>2885</td>
      <td>51</td>
      <td>588</td>
      <td>70</td>
      <td>603</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮,  , 小型犬, 成犬, 通用型, PR27,  , 2KG,  , 博美,...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>9</th>
      <td>531270854040</td>
      <td>Royal Canin皇家狗粮 法国斗牛犬幼犬粮FBJ30/3KG 法斗粮</td>
      <td>法国斗牛犬幼犬粮3KG1套装</td>
      <td>259.0</td>
      <td>1493</td>
      <td>54</td>
      <td>429</td>
      <td>129</td>
      <td>499</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>斗牛犬</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 法国, 斗牛犬, 幼犬粮, FBJ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>10</th>
      <td>26287800319</td>
      <td>Royal Canin皇家狗粮 小型犬幼犬粮MIJ31 8KG 犬主粮28省包邮</td>
      <td>(小型犬)幼犬犬粮 8kg</td>
      <td>420.0</td>
      <td>1819</td>
      <td>60</td>
      <td>510</td>
      <td>210</td>
      <td>459</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>8000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 小型犬, 幼犬粮, MIJ31, ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>11</th>
      <td>19009998266</td>
      <td>皇家繁育期母犬 离乳期幼犬狗粮中型犬奶糕MES30 4KG 松狮哈士奇</td>
      <td>(中型犬)离乳期奶糕 4kg</td>
      <td>302.0</td>
      <td>1200</td>
      <td>47</td>
      <td>274</td>
      <td>151</td>
      <td>392</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>奶糕</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>4000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 繁育, 期母, 犬,  , 离乳期, 幼犬, 狗粮, 中型犬, 奶糕, MES30...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>12</th>
      <td>38792201087</td>
      <td>皇家狗粮 吉娃娃成犬专用粮食C28 1.5KG*2  小型犬 28省包邮</td>
      <td>吉娃娃幼犬套装</td>
      <td>262.0</td>
      <td>1522</td>
      <td>89</td>
      <td>522</td>
      <td>131</td>
      <td>321</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>吉娃娃</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮,  , 吉娃娃, 成犬, 专用, 粮食, C28,  , 1.5, KG, ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>13</th>
      <td>21228115701</td>
      <td>皇家狗粮 小型犬成犬狗粮 通用型狗粮均衡营养减少牙石PR27/0.8KG</td>
      <td>(小型犬)成犬犬粮 800G</td>
      <td>54.0</td>
      <td>2883</td>
      <td>87</td>
      <td>695</td>
      <td>27</td>
      <td>326</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>800</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮,  , 小型犬, 成犬, 狗粮,  , 通用型, 狗粮, 均衡, 营养, 减...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>14</th>
      <td>26985332272</td>
      <td>皇家小型犬幼犬粮MIJ31 2KG博美西施杜宾腊肠犬狗粮通用型全犬种</td>
      <td>(小型犬)幼犬犬粮 2KG</td>
      <td>148.0</td>
      <td>1954</td>
      <td>26</td>
      <td>340</td>
      <td>74</td>
      <td>458</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 小型犬, 幼犬粮, MIJ31,  , 2KG, 博美, 西施, 杜宾, 腊肠犬,...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>15</th>
      <td>21226807286</td>
      <td>Royal Canin皇家狗粮 泰迪/贵宾成犬粮PD30/7.5KG 犬主粮</td>
      <td>贵宾/泰迪(小型犬)成犬专用粮 7.5KG</td>
      <td>504.0</td>
      <td>1815</td>
      <td>97</td>
      <td>552</td>
      <td>252</td>
      <td>560</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>7500</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 泰迪, /, 贵宾, 成犬粮, P...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>16</th>
      <td>19016342292</td>
      <td>皇家狗粮中型犬成犬粮M25 4KG哈士奇松狮萨摩耶粮柯基牛头梗热卖</td>
      <td>(中型犬)成犬犬粮 4kg</td>
      <td>220.0</td>
      <td>2200</td>
      <td>89</td>
      <td>385</td>
      <td>110</td>
      <td>440</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>4000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮, 中型犬, 成犬粮, M25,  , 4KG, 哈士奇, 松狮, 萨摩耶, ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>17</th>
      <td>19010725269</td>
      <td>Roayl Canin皇家狗粮 贵宾幼犬粮 泰迪专用粮APD33/0.5kg 犬主粮</td>
      <td>贵宾/泰迪幼犬专用粮 500G</td>
      <td>52.0</td>
      <td>3539</td>
      <td>79</td>
      <td>646</td>
      <td>26</td>
      <td>746</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>500</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Roayl,  , Canin, 皇家, 狗粮,  , 贵宾, 幼犬粮,  , 泰迪, 专...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>18</th>
      <td>531321268231</td>
      <td>皇家官方旗舰店 贵宾泰迪狗粮8岁以上老年犬PDA26 3KG热卖新品</td>
      <td>贵宾8岁以上老年犬粮3KG1套装</td>
      <td>269.0</td>
      <td>887</td>
      <td>63</td>
      <td>302</td>
      <td>134</td>
      <td>181</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 官方, 旗舰店,  , 贵宾, 泰迪, 狗粮, 8, 岁, 以上, 老年犬, PD...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>19</th>
      <td>35590666167</td>
      <td>皇家狗粮小型犬8岁以上成犬粮SPR27 4KG*2  博美犬茶杯 28省包邮</td>
      <td>SPR27/4KG套装</td>
      <td>486.0</td>
      <td>1609</td>
      <td>77</td>
      <td>557</td>
      <td>243</td>
      <td>2497</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>8000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮, 小型犬, 8, 岁, 以上, 成犬粮, SPR27,  , 4KG, *,...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21979291525</td>
      <td>Royal Canin皇家狗粮 贵宾/泰迪成犬粮PD30/3KG 犬主粮</td>
      <td>贵宾/泰迪(小型犬)成犬专用粮 3KG</td>
      <td>256.0</td>
      <td>4855</td>
      <td>86</td>
      <td>1069</td>
      <td>128</td>
      <td>765</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 贵宾, /, 泰迪, 成犬粮, P...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21224163082</td>
      <td>Royal Canin皇家狗粮 大型犬成犬粮GR26/15KG 犬主粮28省包邮</td>
      <td>(大型犬)成犬犬粮 15KG</td>
      <td>565.0</td>
      <td>6368</td>
      <td>116</td>
      <td>1003</td>
      <td>282</td>
      <td>776</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>15000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 成犬粮, GR26, /...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>22</th>
      <td>39489288478</td>
      <td>皇家官方旗舰店 狗粮比熊犬成犬粮BF29 3KG主粮热卖新品小型犬</td>
      <td>比熊成犬粮BF29/3KG套装</td>
      <td>256.0</td>
      <td>6132</td>
      <td>85</td>
      <td>827</td>
      <td>128</td>
      <td>1171</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>比熊</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 官方, 旗舰店,  , 狗粮, 比熊犬, 成犬粮, BF29,  , 3KG, 主...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>23</th>
      <td>41376458503</td>
      <td>皇家狗粮 中型犬奶糕MES30 4KG*2繁育母犬离乳幼柯基萨摩28省包邮</td>
      <td>中型犬奶糕4KG*2套装</td>
      <td>520.0</td>
      <td>2380</td>
      <td>72</td>
      <td>562</td>
      <td>260</td>
      <td>741</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>奶糕</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>8000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮,  , 中型犬, 奶糕, MES30,  , 4KG, *, 2, 繁育, ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>24</th>
      <td>18995406313</td>
      <td>Royal Canin皇家狗粮 贵宾成犬专用粮PD30/0.5KG 泰迪犬主粮</td>
      <td>贵宾/泰迪(小型犬)成犬专用粮 500G</td>
      <td>49.0</td>
      <td>6320</td>
      <td>98</td>
      <td>1407</td>
      <td>24</td>
      <td>666</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>500</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 贵宾, 成犬, 专用粮, PD30...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>25</th>
      <td>522184641035</td>
      <td>Royal Canin皇家狗粮 大型犬成犬粮GR26/4KG*4包犬主粮 28省包邮</td>
      <td>大型成犬4KG*4套装</td>
      <td>621.0</td>
      <td>6634</td>
      <td>172</td>
      <td>1621</td>
      <td>310</td>
      <td>1740</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>16000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 成犬粮, GR26, /...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>26</th>
      <td>544515300121</td>
      <td>Royal Canin皇家狗粮 金毛幼犬粮AGR29 3.5KG 大型犬狗粮精品热卖</td>
      <td>金毛(大型犬)幼犬 3500g</td>
      <td>229.0</td>
      <td>1094</td>
      <td>93</td>
      <td>264</td>
      <td>114</td>
      <td>671</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>金毛</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3500</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 金毛, 幼犬粮, AGR29,  ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>27</th>
      <td>521992018841</td>
      <td>皇家小型犬幼犬粮通用型狗粮 MIJ31 0.8KG博美八哥腊肠京巴犬热卖</td>
      <td>(小型犬)幼犬犬粮 800G</td>
      <td>61.0</td>
      <td>2736</td>
      <td>93</td>
      <td>775</td>
      <td>30</td>
      <td>534</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>800</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 小型犬, 幼犬粮, 通用型, 狗粮,  , MIJ31,  , 0.8, KG, ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>28</th>
      <td>18953082004</td>
      <td>皇家小型犬成犬粮 通用型狗粮PR27 8KG 博美京巴全犬种28省包邮</td>
      <td>(小型犬)成犬犬粮 8kg</td>
      <td>395.0</td>
      <td>3767</td>
      <td>220</td>
      <td>1189</td>
      <td>197</td>
      <td>761</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>8000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 小型犬, 成犬粮,  , 通用型, 狗粮, PR27,  , 8KG,  , 博美...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>29</th>
      <td>27481836189</td>
      <td>Royal Canin皇家狗粮 约克夏成犬粮PRY28/1.5KG*2犬主粮 28省包邮</td>
      <td>约克夏成犬粮1.5KG *2套装</td>
      <td>242.0</td>
      <td>2503</td>
      <td>149</td>
      <td>746</td>
      <td>121</td>
      <td>339</td>
      <td>其他</td>
      <td>...</td>
      <td>ROYAL CANIN/皇家</td>
      <td>专用粮</td>
      <td>约克夏梗</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 约克夏, 成犬粮, PRY28, ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153</th>
      <td>38467261575</td>
      <td>比瑞吉小型成犬粮 小型犬成犬天然粮 比熊泰迪成犬比瑞吉狗粮10kg</td>
      <td>(小型犬)成犬海藻胡萝卜粒配方犬粮 10KG</td>
      <td>439.0</td>
      <td>34516</td>
      <td>1139</td>
      <td>9119</td>
      <td>219</td>
      <td>9190</td>
      <td>其他</td>
      <td>...</td>
      <td>Nature Bridge/比瑞吉</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>比瑞吉宠物用品股份有限公司</td>
      <td>10000</td>
      <td>中国</td>
      <td>上海市金山区</td>
      <td>海藻胡萝卜粒配方</td>
      <td>[比瑞吉, 小型, 成犬粮,  , 小型犬, 成犬, 天然粮,  , 比熊, 泰迪, 成犬,...</td>
      <td>营养均衡配方</td>
    </tr>
    <tr>
      <th>154</th>
      <td>521252183250</td>
      <td>比瑞吉无谷六种肉全期犬粮 非五谷狗粮 无谷多肉成犬幼犬狗粮2kg</td>
      <td>全犬期六种肉配方犬粮 2000g</td>
      <td>199.0</td>
      <td>27034</td>
      <td>1158</td>
      <td>7897</td>
      <td>99</td>
      <td>7312</td>
      <td>其他</td>
      <td>...</td>
      <td>Nature Bridge/比瑞吉</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>比瑞吉宠物用品股份有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海市金山区</td>
      <td>无谷六种肉配方</td>
      <td>[比瑞吉, 无谷, 六种, 肉, 全期犬粮,  , 非, 五谷, 狗粮,  , 无谷, 多肉...</td>
      <td>无谷多肉配方</td>
    </tr>
    <tr>
      <th>155</th>
      <td>38444695810</td>
      <td>比瑞吉狗粮 小型成犬粮通用型成犬狗粮2kg 雪纳瑞博美泰迪狗粮</td>
      <td>(小型犬)成犬犬粮 2KG</td>
      <td>120.0</td>
      <td>49771</td>
      <td>3847</td>
      <td>11093</td>
      <td>60</td>
      <td>4730</td>
      <td>其他</td>
      <td>...</td>
      <td>Nature Bridge/比瑞吉</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>比瑞吉宠物用品股份有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海市金山区亭卫公路</td>
      <td>无</td>
      <td>[比瑞吉, 狗粮,  , 小型, 成犬粮, 通用型, 成犬, 狗粮, 2kg,  , 雪纳瑞...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>156</th>
      <td>38442639432</td>
      <td>狗粮 比瑞吉泰迪贵宾成犬粮 泰迪狗粮成犬天然粮 比瑞吉狗粮2kg</td>
      <td>贵宾/泰迪(小型犬)成犬专用粮 2KG</td>
      <td>138.0</td>
      <td>337524</td>
      <td>8018</td>
      <td>56434</td>
      <td>69</td>
      <td>34370</td>
      <td>其他</td>
      <td>...</td>
      <td>Nature Bridge/比瑞吉</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>比瑞吉宠物用品股份有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海市金山区亭卫公路</td>
      <td>无</td>
      <td>[狗粮,  , 比瑞吉, 泰迪, 贵宾, 成犬粮,  , 泰迪, 狗粮, 成犬, 天然粮, ...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>157</th>
      <td>38453114132</td>
      <td>比瑞吉狗粮 小型成犬粮泰迪贵宾比熊博美雪纳瑞通用天然狗粮1.5kg</td>
      <td>(小型犬)成犬海藻胡萝卜粒配方犬粮 1.5KG</td>
      <td>90.0</td>
      <td>81282</td>
      <td>1255</td>
      <td>16747</td>
      <td>45</td>
      <td>12084</td>
      <td>其他</td>
      <td>...</td>
      <td>Nature Bridge/比瑞吉</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>上海比瑞吉宠物用品股份有限公司</td>
      <td>1500</td>
      <td>中国</td>
      <td>上海市金山区</td>
      <td>海藻胡萝卜粒配方</td>
      <td>[比瑞吉, 狗粮,  , 小型, 成犬粮, 泰迪, 贵宾, 比熊博美, 雪纳瑞, 通用, 天...</td>
      <td>营养均衡配方</td>
    </tr>
    <tr>
      <th>158</th>
      <td>38463401287</td>
      <td>狗粮幼犬 比瑞吉小型幼犬粮 泰迪比熊通用 室内小型全价幼犬粮2kg</td>
      <td>(小型犬)幼犬犬粮 2KG</td>
      <td>118.0</td>
      <td>39075</td>
      <td>1852</td>
      <td>9378</td>
      <td>59</td>
      <td>6548</td>
      <td>其他</td>
      <td>...</td>
      <td>Nature Bridge/比瑞吉</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>比瑞吉宠物用品股份有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海市金山区亭卫公路</td>
      <td>无</td>
      <td>[狗粮, 幼犬,  , 比瑞吉, 小型, 幼犬粮,  , 泰迪, 比熊, 通用,  , 室内...</td>
      <td>无</td>
    </tr>
    <tr>
      <th>159</th>
      <td>564032233487</td>
      <td>海洋之星SUPERIOR进口营养加强体重控制三文鱼狗粮泰迪通用型12kg</td>
      <td>全犬期精选海洋营养加强体重控制三文鱼配方犬粮 12000g</td>
      <td>999.0</td>
      <td>27</td>
      <td>12</td>
      <td>9</td>
      <td>499</td>
      <td>30</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>12000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>精选海洋营养加强体重控制三文鱼配方</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 体重, 控制, 三文鱼, 狗粮, ...</td>
      <td>三文鱼味配方</td>
    </tr>
    <tr>
      <th>160</th>
      <td>564179827386</td>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼成犬狗粮泰迪通用型12kg</td>
      <td>成犬精选海洋营养加强三文鱼成犬配方犬粮 12000g</td>
      <td>930.0</td>
      <td>60</td>
      <td>23</td>
      <td>22</td>
      <td>465</td>
      <td>51</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>12000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>精选海洋营养加强三文鱼成犬配方</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, ...</td>
      <td>三文鱼味配方</td>
    </tr>
    <tr>
      <th>161</th>
      <td>550578330260</td>
      <td>海洋之星SUPERIOR进口营养加强体重控制三文鱼狗粮中大型犬12kg</td>
      <td>成犬Superior（优级）三文鱼控制体重配方大颗粒12kg犬粮 12000g</td>
      <td>999.0</td>
      <td>77</td>
      <td>26</td>
      <td>35</td>
      <td>499</td>
      <td>35</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>12000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>Superior（优级）三文鱼控制体重配方大颗粒12kg</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 体重, 控制, 三文鱼, 狗粮, ...</td>
      <td>三文鱼味配方</td>
    </tr>
    <tr>
      <th>162</th>
      <td>550548533749</td>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼成犬狗粮中大型犬12kg</td>
      <td>成犬鱼肉味犬粮 12000g</td>
      <td>930.0</td>
      <td>104</td>
      <td>21</td>
      <td>29</td>
      <td>465</td>
      <td>37</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>12000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>鱼肉味</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, ...</td>
      <td>鱼肉口味配方</td>
    </tr>
    <tr>
      <th>163</th>
      <td>564102078063</td>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼幼犬狗粮中大型犬12kg</td>
      <td>(中大型犬)幼犬精选海洋营养加强三文鱼幼犬配方犬粮 12000g</td>
      <td>970.0</td>
      <td>20</td>
      <td>9</td>
      <td>4</td>
      <td>485</td>
      <td>34</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>12000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>精选海洋营养加强三文鱼幼犬配方</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 幼犬, 狗粮, ...</td>
      <td>三文鱼味配方</td>
    </tr>
    <tr>
      <th>164</th>
      <td>543492643164</td>
      <td>海洋之星狗粮金毛阿拉斯加大型犬幼犬无谷天然粮大颗粒犬粮12kg</td>
      <td>(中大型犬)幼犬犬粮 12000g</td>
      <td>750.0</td>
      <td>1936</td>
      <td>100</td>
      <td>539</td>
      <td>375</td>
      <td>689</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>鱼肉味</td>
      <td>[海洋之星, 狗粮, 金毛, 阿拉斯加, 大型犬, 幼犬, 无谷, 天然粮, 大颗粒, 犬粮...</td>
      <td>鱼肉口味配方</td>
    </tr>
    <tr>
      <th>165</th>
      <td>550575110336</td>
      <td>海洋之星SUPERIOR原装进口营养加强体重控制三文鱼配方狗粮6kg</td>
      <td>成犬Superior（优级）三文鱼控制体重小颗粒6kg犬粮 6000g</td>
      <td>570.0</td>
      <td>140</td>
      <td>48</td>
      <td>52</td>
      <td>285</td>
      <td>66</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>6000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>Superior（优级）三文鱼控制体重小颗粒6kg</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 体重, 控制, 三文鱼, ...</td>
      <td>三文鱼味配方</td>
    </tr>
    <tr>
      <th>166</th>
      <td>548685638984</td>
      <td>海洋之星SUPERIOR进口营养加强体重控制三文鱼狗粮通用型1.5kg</td>
      <td>成犬Superior（优级）体重控制小颗粒1.5kg犬粮 1500g</td>
      <td>230.0</td>
      <td>496</td>
      <td>58</td>
      <td>166</td>
      <td>115</td>
      <td>65</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>1500</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>Superior（优级）体重控制小颗粒1.5kg</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 体重, 控制, 三文鱼, 狗粮, ...</td>
      <td>鱼肉口味配方</td>
    </tr>
    <tr>
      <th>167</th>
      <td>550550421444</td>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼幼犬狗粮泰迪通用型6kg</td>
      <td>幼犬三文鱼犬粮 6000g</td>
      <td>550.0</td>
      <td>175</td>
      <td>29</td>
      <td>55</td>
      <td>275</td>
      <td>56</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>6000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>三文鱼</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 幼犬, 狗粮, ...</td>
      <td>三文鱼味配方</td>
    </tr>
    <tr>
      <th>168</th>
      <td>550572214265</td>
      <td>海洋之星SUPERIOR进口营养加强三文鱼成犬狗粮泰迪通用型1.5kg</td>
      <td>(中小型犬)成犬三文鱼犬粮 1500g</td>
      <td>190.0</td>
      <td>1048</td>
      <td>203</td>
      <td>337</td>
      <td>95</td>
      <td>135</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>1500</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>三文鱼</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, 泰迪, ...</td>
      <td>三文鱼味配方</td>
    </tr>
    <tr>
      <th>169</th>
      <td>534010155073</td>
      <td>大幼试吃装中大型犬幼犬金毛阿拉斯牧羊犬哈士奇狗粮三包*30g</td>
      <td>(大型犬)幼犬深海鱼配方犬粮 100g</td>
      <td>9.9</td>
      <td>7543</td>
      <td>206</td>
      <td>1836</td>
      <td>4</td>
      <td>476</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>海洋之星</td>
      <td>100</td>
      <td>其他</td>
      <td>Fish4Dogs.Ltd</td>
      <td>深海鱼配方</td>
      <td>[大幼, 试吃装, 中大型犬, 幼犬, 金毛, 阿拉斯, 牧羊犬, 哈士奇, 狗粮, 三包,...</td>
      <td>深海鱼味配方</td>
    </tr>
    <tr>
      <th>170</th>
      <td>550574250453</td>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼成犬狗粮泰迪通用型6kg</td>
      <td>成犬鱼肉犬粮 6000g</td>
      <td>520.0</td>
      <td>278</td>
      <td>85</td>
      <td>76</td>
      <td>260</td>
      <td>85</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>6000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>鱼肉</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, ...</td>
      <td>鱼肉口味配方</td>
    </tr>
    <tr>
      <th>171</th>
      <td>534139840517</td>
      <td>试吃装 深海鱼三文鱼成犬中大型犬试吃装大颗粒阿拉斯加三包*30g</td>
      <td>(大型犬)成犬鱼肉味犬粮 100g</td>
      <td>9.9</td>
      <td>7029</td>
      <td>238</td>
      <td>1519</td>
      <td>4</td>
      <td>247</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>Fish4Dogs.Ltd</td>
      <td>100</td>
      <td>其他</td>
      <td>Fish4Dogs.Ltd</td>
      <td>鱼肉味</td>
      <td>[试吃装,  , 深海鱼, 三文鱼, 成犬, 中大型犬, 试吃装, 大颗粒, 阿拉斯加, 三...</td>
      <td>鱼肉口味配方</td>
    </tr>
    <tr>
      <th>172</th>
      <td>529421904631</td>
      <td>海洋之星深海鱼狗粮无谷天然粮中小型犬成犬主粮小颗粒6kg</td>
      <td>成犬无谷物深海鱼小颗粒配方犬粮 其它</td>
      <td>410.0</td>
      <td>5723</td>
      <td>275</td>
      <td>2018</td>
      <td>205</td>
      <td>717</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>6000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物深海鱼小颗粒配方</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 中小型犬, 成, 犬主粮, 小颗粒, 6kg]</td>
      <td>无谷深海鱼配方</td>
    </tr>
    <tr>
      <th>173</th>
      <td>521099649915</td>
      <td>海洋之星深海鱼狗粮无谷天然粮金毛萨摩耶大型犬成犬大颗粒12kg</td>
      <td>成犬无谷物深海鱼大颗粒配方犬粮 12kg</td>
      <td>710.0</td>
      <td>6287</td>
      <td>311</td>
      <td>1950</td>
      <td>355</td>
      <td>929</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物深海鱼大颗粒配方</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 金毛, 萨摩耶, 大型犬, 成犬, 大颗粒...</td>
      <td>无谷深海鱼配方</td>
    </tr>
    <tr>
      <th>174</th>
      <td>521103914337</td>
      <td>海洋之星深海鱼狗粮无谷天然粮中小型犬幼犬小颗粒狗粮6kg</td>
      <td>幼犬无谷物深海鱼小颗粒6kg犬粮</td>
      <td>440.0</td>
      <td>7869</td>
      <td>245</td>
      <td>2504</td>
      <td>220</td>
      <td>1578</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>6000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>幼犬无谷物深海鱼小颗粒6kg</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 中小型犬, 幼犬, 小颗粒, 狗粮, 6kg]</td>
      <td>无谷深海鱼配方</td>
    </tr>
    <tr>
      <th>175</th>
      <td>521091943822</td>
      <td>海洋之星深海鱼狗粮无谷天然粮中小型犬幼犬小颗粒12kg</td>
      <td>幼犬无谷物深海鱼幼犬配方小颗粒犬粮 12kg</td>
      <td>750.0</td>
      <td>7739</td>
      <td>264</td>
      <td>2230</td>
      <td>375</td>
      <td>2027</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物深海鱼幼犬配方小颗粒</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 中小型犬, 幼犬, 小颗粒, 12kg]</td>
      <td>无谷深海鱼配方</td>
    </tr>
    <tr>
      <th>176</th>
      <td>521092399322</td>
      <td>海洋之星狗粮三文鱼无谷天然粮金毛拉布拉多大型犬成犬主粮 12kg</td>
      <td>成犬无谷物三文鱼大颗粒配方犬粮 12kg</td>
      <td>710.0</td>
      <td>10751</td>
      <td>452</td>
      <td>2668</td>
      <td>355</td>
      <td>2130</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物三文鱼大颗粒配方</td>
      <td>[海洋之星, 狗粮, 三文鱼, 无谷, 天然粮, 金毛, 拉布拉多, 大型犬, 成, 犬主粮...</td>
      <td>无谷三文鱼配方</td>
    </tr>
    <tr>
      <th>177</th>
      <td>521323201615</td>
      <td>海洋之星深海鱼成犬狗粮无谷天然粮中小型犬成犬小颗粒1.5kg</td>
      <td>成犬无谷物深海鱼小颗粒1.5kg犬粮</td>
      <td>150.0</td>
      <td>10626</td>
      <td>352</td>
      <td>3402</td>
      <td>75</td>
      <td>1338</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>1500</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>成犬无谷物深海鱼小颗粒1.5kg</td>
      <td>[海洋之星, 深海鱼, 成犬, 狗粮, 无谷, 天然粮, 中小型犬, 成犬, 小颗粒, 1....</td>
      <td>无谷深海鱼配方</td>
    </tr>
    <tr>
      <th>178</th>
      <td>521103456733</td>
      <td>海洋之星深海鱼成犬狗粮无谷天然粮中小型犬成犬狗粮12kg小颗粒</td>
      <td>成犬无谷物深海鱼大颗粒配方犬粮 其它</td>
      <td>710.0</td>
      <td>7173</td>
      <td>388</td>
      <td>2226</td>
      <td>355</td>
      <td>1171</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>6000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物深海鱼大颗粒配方</td>
      <td>[海洋之星, 深海鱼, 成犬, 狗粮, 无谷, 天然粮, 中小型犬, 成犬, 狗粮, 12k...</td>
      <td>无谷深海鱼配方</td>
    </tr>
    <tr>
      <th>179</th>
      <td>545052402563</td>
      <td>海洋之星狗粮试吃装泰迪比熊玩具犬幼犬成犬全犬期小颗粒三包*30g</td>
      <td>全犬期鱼味犬粮 100g</td>
      <td>9.9</td>
      <td>10623</td>
      <td>349</td>
      <td>3003</td>
      <td>12</td>
      <td>578</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>海洋之星</td>
      <td>100</td>
      <td>其他</td>
      <td>Fish4Dogs.Ltd</td>
      <td>鱼味</td>
      <td>[海洋之星, 狗粮, 试吃装, 泰迪, 比熊, 玩具犬, 幼犬, 成犬, 全犬期, 小颗粒,...</td>
      <td>鱼肉口味配方</td>
    </tr>
    <tr>
      <th>180</th>
      <td>521103920550</td>
      <td>海洋之星狗粮通用型天然粮中小型犬幼犬主粮泰迪博美比熊1.5kg</td>
      <td>幼犬无谷物深海鱼幼犬配方大颗粒犬粮 1.5kg</td>
      <td>160.0</td>
      <td>25402</td>
      <td>614</td>
      <td>7582</td>
      <td>80</td>
      <td>7682</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>1500</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物深海鱼幼犬配方大颗粒</td>
      <td>[海洋之星, 狗粮, 通用型, 天然粮, 中小型犬, 幼犬主粮, 泰迪, 博美比, 熊, 1...</td>
      <td>无谷深海鱼配方</td>
    </tr>
    <tr>
      <th>181</th>
      <td>534102297282</td>
      <td>海洋之星狗粮试吃装泰迪比熊贵宾幼犬通用型小颗粒美毛三包*30g</td>
      <td>(中小型犬)幼犬深海鱼配方犬粮 100g</td>
      <td>9.9</td>
      <td>30387</td>
      <td>1018</td>
      <td>8754</td>
      <td>4</td>
      <td>2003</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>海洋之星</td>
      <td>100</td>
      <td>其他</td>
      <td>Fish4Dogs.Ltd</td>
      <td>深海鱼配方</td>
      <td>[海洋之星, 狗粮, 试吃装, 泰迪, 比熊, 贵宾, 幼犬, 通用型, 小颗粒, 美毛, ...</td>
      <td>深海鱼味配方</td>
    </tr>
    <tr>
      <th>182</th>
      <td>521100097514</td>
      <td>海洋之星三文鱼狗粮泰迪吉娃娃贵宾玩具犬狗粮无谷天然粮6kg</td>
      <td>贵宾/泰迪无谷物三文鱼玩具犬配方犬粮</td>
      <td>440.0</td>
      <td>8286</td>
      <td>573</td>
      <td>2659</td>
      <td>220</td>
      <td>1750</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>Fish4Dogs/海洋之星</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>山东海创工贸有限公司</td>
      <td>6000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物三文鱼玩具犬配方</td>
      <td>[海洋之星, 三文鱼, 狗粮, 泰迪, 吉娃娃, 贵宾, 玩具犬, 狗粮, 无谷, 天然粮,...</td>
      <td>无谷三文鱼配方</td>
    </tr>
  </tbody>
</table>
<p>183 rows × 22 columns</p>

</div>



### 品牌

```python
df_raw_1.Brand.value_counts(dropna=False)
```



```
ROYAL CANIN/皇家       91
Pedigree/宝路          38
Nature Bridge/比瑞吉    28
Fish4Dogs/海洋之星       24
Chappi/佳贝             2
Name: Brand, dtype: int64
```



```python
brand_2 = []
for i in range(len(df_raw_1)):
    if df_raw_1.Brand[i] == 'Pedigree/宝路' or df_raw_1.Brand[i] == 'Chappi/佳贝' :
        brand_2.append('玛氏')
    elif df_raw_1.Brand[i] == 'ROYAL CANIN/皇家':
        brand_2.append('皇家')
    elif df_raw_1.Brand[i] == 'Nature Bridge/比瑞吉':
        brand_2.append('比瑞吉')
    elif df_raw_1.Brand[i] == 'Fish4Dogs/海洋之星':
        brand_2.append('海洋之星')        
```

```python
df_raw_1['Brand_v2']= brand_2
```

```python
df_raw_1
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_id</th>
      <th>item_name</th>
      <th>TradeName</th>
      <th>price</th>
      <th>total_sale</th>
      <th>month_sale</th>
      <th>accum_comm</th>
      <th>TM_points</th>
      <th>CollectCount</th>
      <th>Tastes</th>
      <th>...</th>
      <th>Classification</th>
      <th>Breed</th>
      <th>Manufacturer</th>
      <th>Weight</th>
      <th>Origin</th>
      <th>ManufacturerAddress</th>
      <th>RecipeTastePrescription</th>
      <th>item_name_cut</th>
      <th>RecipeTastePrescription_v2</th>
      <th>Brand_v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19045209534</td>
      <td>Royal Canin皇家狗粮 德牧幼犬粮AGS30 12KG 大型犬狗粮28省包邮</td>
      <td>德国牧羊犬幼犬专用粮 12kg</td>
      <td>650.0</td>
      <td>2046</td>
      <td>39</td>
      <td>418</td>
      <td>325</td>
      <td>348</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>德国牧羊犬</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 德牧, 幼犬粮, AGS30,  ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>1</th>
      <td>547144906363</td>
      <td>Royal Canin皇家狗粮 柴犬幼犬专用粮SIJ29/3KG 犬主粮狗粮</td>
      <td>日本柴犬幼犬 3000g</td>
      <td>285.0</td>
      <td>560</td>
      <td>39</td>
      <td>132</td>
      <td>142</td>
      <td>200</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>日本柴犬</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 柴犬, 幼犬, 专用粮, SIJ2...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26255560615</td>
      <td>皇家狗粮 大型犬奶糕MAS30 4KG哺乳孕期犬离乳期幼犬牧羊阿拉斯加</td>
      <td>(大型犬)离乳期奶糕 4kg</td>
      <td>301.0</td>
      <td>1571</td>
      <td>27</td>
      <td>390</td>
      <td>150</td>
      <td>554</td>
      <td>其他</td>
      <td>...</td>
      <td>奶糕</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>4000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮,  , 大型犬, 奶糕, MAS30,  , 4KG, 哺乳, 孕期, 犬离...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39792870853</td>
      <td>Royal Canin皇家狗粮 迷你雪纳瑞成犬粮SNZ25/3KG 犬主粮</td>
      <td>雪纳瑞成犬 3000g</td>
      <td>260.0</td>
      <td>2113</td>
      <td>32</td>
      <td>281</td>
      <td>130</td>
      <td>379</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>雪纳瑞</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海奉贤区肖南路475号</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 迷你, 雪纳瑞, 成犬粮, SNZ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>4</th>
      <td>547165690913</td>
      <td>Royal Canin皇家狗粮 拉布拉多幼犬粮ALR33 3KG 大型犬全犬种热卖</td>
      <td>拉布拉多幼犬 3000g</td>
      <td>210.0</td>
      <td>643</td>
      <td>48</td>
      <td>173</td>
      <td>105</td>
      <td>318</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>拉布拉多</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 拉布拉多, 幼犬粮, ALR33,...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>5</th>
      <td>19086681580</td>
      <td>Royal Canin皇家狗粮 小型犬老年犬狗粮SPR27/0.8kg公斤  犬主粮</td>
      <td>(小型犬)老年犬犬粮 800G</td>
      <td>66.0</td>
      <td>1906</td>
      <td>46</td>
      <td>473</td>
      <td>33</td>
      <td>205</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>800</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 小型犬, 老年犬, 狗粮, SPR...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>6</th>
      <td>18965425186</td>
      <td>Royal Canin皇家狗粮 大型犬幼犬粮MAJ30/15KG  犬主粮28省包邮</td>
      <td>(大型犬)幼犬犬粮 15KG</td>
      <td>623.0</td>
      <td>5018</td>
      <td>56</td>
      <td>771</td>
      <td>311</td>
      <td>1163</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>15000</td>
      <td>中国</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 幼犬粮, MAJ30, ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>7</th>
      <td>26217124504</td>
      <td>Royal Canin皇家狗粮 约克夏成犬专用粮PRY28/1.5KG 犬主粮</td>
      <td>约克夏梗(小型犬)成犬专用粮 1.5KG</td>
      <td>135.0</td>
      <td>2057</td>
      <td>31</td>
      <td>329</td>
      <td>67</td>
      <td>238</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>约克夏梗</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>1500</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 约克夏, 成犬, 专用粮, PRY...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>8</th>
      <td>19738658492</td>
      <td>皇家狗粮 小型犬成犬通用型PR27 2KG 博美迷你腊肠京巴八哥全犬种</td>
      <td>(小型犬)成犬犬粮 2000g</td>
      <td>141.0</td>
      <td>2885</td>
      <td>51</td>
      <td>588</td>
      <td>70</td>
      <td>603</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮,  , 小型犬, 成犬, 通用型, PR27,  , 2KG,  , 博美,...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>9</th>
      <td>531270854040</td>
      <td>Royal Canin皇家狗粮 法国斗牛犬幼犬粮FBJ30/3KG 法斗粮</td>
      <td>法国斗牛犬幼犬粮3KG1套装</td>
      <td>259.0</td>
      <td>1493</td>
      <td>54</td>
      <td>429</td>
      <td>129</td>
      <td>499</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>斗牛犬</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 法国, 斗牛犬, 幼犬粮, FBJ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>10</th>
      <td>26287800319</td>
      <td>Royal Canin皇家狗粮 小型犬幼犬粮MIJ31 8KG 犬主粮28省包邮</td>
      <td>(小型犬)幼犬犬粮 8kg</td>
      <td>420.0</td>
      <td>1819</td>
      <td>60</td>
      <td>510</td>
      <td>210</td>
      <td>459</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>8000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 小型犬, 幼犬粮, MIJ31, ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>11</th>
      <td>19009998266</td>
      <td>皇家繁育期母犬 离乳期幼犬狗粮中型犬奶糕MES30 4KG 松狮哈士奇</td>
      <td>(中型犬)离乳期奶糕 4kg</td>
      <td>302.0</td>
      <td>1200</td>
      <td>47</td>
      <td>274</td>
      <td>151</td>
      <td>392</td>
      <td>其他</td>
      <td>...</td>
      <td>奶糕</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>4000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 繁育, 期母, 犬,  , 离乳期, 幼犬, 狗粮, 中型犬, 奶糕, MES30...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>12</th>
      <td>38792201087</td>
      <td>皇家狗粮 吉娃娃成犬专用粮食C28 1.5KG*2  小型犬 28省包邮</td>
      <td>吉娃娃幼犬套装</td>
      <td>262.0</td>
      <td>1522</td>
      <td>89</td>
      <td>522</td>
      <td>131</td>
      <td>321</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>吉娃娃</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮,  , 吉娃娃, 成犬, 专用, 粮食, C28,  , 1.5, KG, ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>13</th>
      <td>21228115701</td>
      <td>皇家狗粮 小型犬成犬狗粮 通用型狗粮均衡营养减少牙石PR27/0.8KG</td>
      <td>(小型犬)成犬犬粮 800G</td>
      <td>54.0</td>
      <td>2883</td>
      <td>87</td>
      <td>695</td>
      <td>27</td>
      <td>326</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>800</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮,  , 小型犬, 成犬, 狗粮,  , 通用型, 狗粮, 均衡, 营养, 减...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>14</th>
      <td>26985332272</td>
      <td>皇家小型犬幼犬粮MIJ31 2KG博美西施杜宾腊肠犬狗粮通用型全犬种</td>
      <td>(小型犬)幼犬犬粮 2KG</td>
      <td>148.0</td>
      <td>1954</td>
      <td>26</td>
      <td>340</td>
      <td>74</td>
      <td>458</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 小型犬, 幼犬粮, MIJ31,  , 2KG, 博美, 西施, 杜宾, 腊肠犬,...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>15</th>
      <td>21226807286</td>
      <td>Royal Canin皇家狗粮 泰迪/贵宾成犬粮PD30/7.5KG 犬主粮</td>
      <td>贵宾/泰迪(小型犬)成犬专用粮 7.5KG</td>
      <td>504.0</td>
      <td>1815</td>
      <td>97</td>
      <td>552</td>
      <td>252</td>
      <td>560</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>7500</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 泰迪, /, 贵宾, 成犬粮, P...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>16</th>
      <td>19016342292</td>
      <td>皇家狗粮中型犬成犬粮M25 4KG哈士奇松狮萨摩耶粮柯基牛头梗热卖</td>
      <td>(中型犬)成犬犬粮 4kg</td>
      <td>220.0</td>
      <td>2200</td>
      <td>89</td>
      <td>385</td>
      <td>110</td>
      <td>440</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>4000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮, 中型犬, 成犬粮, M25,  , 4KG, 哈士奇, 松狮, 萨摩耶, ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>17</th>
      <td>19010725269</td>
      <td>Roayl Canin皇家狗粮 贵宾幼犬粮 泰迪专用粮APD33/0.5kg 犬主粮</td>
      <td>贵宾/泰迪幼犬专用粮 500G</td>
      <td>52.0</td>
      <td>3539</td>
      <td>79</td>
      <td>646</td>
      <td>26</td>
      <td>746</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>500</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Roayl,  , Canin, 皇家, 狗粮,  , 贵宾, 幼犬粮,  , 泰迪, 专...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>18</th>
      <td>531321268231</td>
      <td>皇家官方旗舰店 贵宾泰迪狗粮8岁以上老年犬PDA26 3KG热卖新品</td>
      <td>贵宾8岁以上老年犬粮3KG1套装</td>
      <td>269.0</td>
      <td>887</td>
      <td>63</td>
      <td>302</td>
      <td>134</td>
      <td>181</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 官方, 旗舰店,  , 贵宾, 泰迪, 狗粮, 8, 岁, 以上, 老年犬, PD...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>19</th>
      <td>35590666167</td>
      <td>皇家狗粮小型犬8岁以上成犬粮SPR27 4KG*2  博美犬茶杯 28省包邮</td>
      <td>SPR27/4KG套装</td>
      <td>486.0</td>
      <td>1609</td>
      <td>77</td>
      <td>557</td>
      <td>243</td>
      <td>2497</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>8000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮, 小型犬, 8, 岁, 以上, 成犬粮, SPR27,  , 4KG, *,...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21979291525</td>
      <td>Royal Canin皇家狗粮 贵宾/泰迪成犬粮PD30/3KG 犬主粮</td>
      <td>贵宾/泰迪(小型犬)成犬专用粮 3KG</td>
      <td>256.0</td>
      <td>4855</td>
      <td>86</td>
      <td>1069</td>
      <td>128</td>
      <td>765</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 贵宾, /, 泰迪, 成犬粮, P...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21224163082</td>
      <td>Royal Canin皇家狗粮 大型犬成犬粮GR26/15KG 犬主粮28省包邮</td>
      <td>(大型犬)成犬犬粮 15KG</td>
      <td>565.0</td>
      <td>6368</td>
      <td>116</td>
      <td>1003</td>
      <td>282</td>
      <td>776</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>15000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 成犬粮, GR26, /...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>22</th>
      <td>39489288478</td>
      <td>皇家官方旗舰店 狗粮比熊犬成犬粮BF29 3KG主粮热卖新品小型犬</td>
      <td>比熊成犬粮BF29/3KG套装</td>
      <td>256.0</td>
      <td>6132</td>
      <td>85</td>
      <td>827</td>
      <td>128</td>
      <td>1171</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>比熊</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 官方, 旗舰店,  , 狗粮, 比熊犬, 成犬粮, BF29,  , 3KG, 主...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>23</th>
      <td>41376458503</td>
      <td>皇家狗粮 中型犬奶糕MES30 4KG*2繁育母犬离乳幼柯基萨摩28省包邮</td>
      <td>中型犬奶糕4KG*2套装</td>
      <td>520.0</td>
      <td>2380</td>
      <td>72</td>
      <td>562</td>
      <td>260</td>
      <td>741</td>
      <td>其他</td>
      <td>...</td>
      <td>奶糕</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>8000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 狗粮,  , 中型犬, 奶糕, MES30,  , 4KG, *, 2, 繁育, ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>24</th>
      <td>18995406313</td>
      <td>Royal Canin皇家狗粮 贵宾成犬专用粮PD30/0.5KG 泰迪犬主粮</td>
      <td>贵宾/泰迪(小型犬)成犬专用粮 500G</td>
      <td>49.0</td>
      <td>6320</td>
      <td>98</td>
      <td>1407</td>
      <td>24</td>
      <td>666</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>500</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 贵宾, 成犬, 专用粮, PD30...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>25</th>
      <td>522184641035</td>
      <td>Royal Canin皇家狗粮 大型犬成犬粮GR26/4KG*4包犬主粮 28省包邮</td>
      <td>大型成犬4KG*4套装</td>
      <td>621.0</td>
      <td>6634</td>
      <td>172</td>
      <td>1621</td>
      <td>310</td>
      <td>1740</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>16000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 大型犬, 成犬粮, GR26, /...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>26</th>
      <td>544515300121</td>
      <td>Royal Canin皇家狗粮 金毛幼犬粮AGR29 3.5KG 大型犬狗粮精品热卖</td>
      <td>金毛(大型犬)幼犬 3500g</td>
      <td>229.0</td>
      <td>1094</td>
      <td>93</td>
      <td>264</td>
      <td>114</td>
      <td>671</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>金毛</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3500</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 金毛, 幼犬粮, AGR29,  ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>27</th>
      <td>521992018841</td>
      <td>皇家小型犬幼犬粮通用型狗粮 MIJ31 0.8KG博美八哥腊肠京巴犬热卖</td>
      <td>(小型犬)幼犬犬粮 800G</td>
      <td>61.0</td>
      <td>2736</td>
      <td>93</td>
      <td>775</td>
      <td>30</td>
      <td>534</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>800</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 小型犬, 幼犬粮, 通用型, 狗粮,  , MIJ31,  , 0.8, KG, ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>28</th>
      <td>18953082004</td>
      <td>皇家小型犬成犬粮 通用型狗粮PR27 8KG 博美京巴全犬种28省包邮</td>
      <td>(小型犬)成犬犬粮 8kg</td>
      <td>395.0</td>
      <td>3767</td>
      <td>220</td>
      <td>1189</td>
      <td>197</td>
      <td>761</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>8000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[皇家, 小型犬, 成犬粮,  , 通用型, 狗粮, PR27,  , 8KG,  , 博美...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>29</th>
      <td>27481836189</td>
      <td>Royal Canin皇家狗粮 约克夏成犬粮PRY28/1.5KG*2犬主粮 28省包邮</td>
      <td>约克夏成犬粮1.5KG *2套装</td>
      <td>242.0</td>
      <td>2503</td>
      <td>149</td>
      <td>746</td>
      <td>121</td>
      <td>339</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>约克夏梗</td>
      <td>皇誉宠物食品（上海）有限公司</td>
      <td>3000</td>
      <td>中国</td>
      <td>上海</td>
      <td>无</td>
      <td>[Royal,  , Canin, 皇家, 狗粮,  , 约克夏, 成犬粮, PRY28, ...</td>
      <td>无</td>
      <td>皇家</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153</th>
      <td>38467261575</td>
      <td>比瑞吉小型成犬粮 小型犬成犬天然粮 比熊泰迪成犬比瑞吉狗粮10kg</td>
      <td>(小型犬)成犬海藻胡萝卜粒配方犬粮 10KG</td>
      <td>439.0</td>
      <td>34516</td>
      <td>1139</td>
      <td>9119</td>
      <td>219</td>
      <td>9190</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>比瑞吉宠物用品股份有限公司</td>
      <td>10000</td>
      <td>中国</td>
      <td>上海市金山区</td>
      <td>海藻胡萝卜粒配方</td>
      <td>[比瑞吉, 小型, 成犬粮,  , 小型犬, 成犬, 天然粮,  , 比熊, 泰迪, 成犬,...</td>
      <td>营养均衡配方</td>
      <td>比瑞吉</td>
    </tr>
    <tr>
      <th>154</th>
      <td>521252183250</td>
      <td>比瑞吉无谷六种肉全期犬粮 非五谷狗粮 无谷多肉成犬幼犬狗粮2kg</td>
      <td>全犬期六种肉配方犬粮 2000g</td>
      <td>199.0</td>
      <td>27034</td>
      <td>1158</td>
      <td>7897</td>
      <td>99</td>
      <td>7312</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>比瑞吉宠物用品股份有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海市金山区</td>
      <td>无谷六种肉配方</td>
      <td>[比瑞吉, 无谷, 六种, 肉, 全期犬粮,  , 非, 五谷, 狗粮,  , 无谷, 多肉...</td>
      <td>无谷多肉配方</td>
      <td>比瑞吉</td>
    </tr>
    <tr>
      <th>155</th>
      <td>38444695810</td>
      <td>比瑞吉狗粮 小型成犬粮通用型成犬狗粮2kg 雪纳瑞博美泰迪狗粮</td>
      <td>(小型犬)成犬犬粮 2KG</td>
      <td>120.0</td>
      <td>49771</td>
      <td>3847</td>
      <td>11093</td>
      <td>60</td>
      <td>4730</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>比瑞吉宠物用品股份有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海市金山区亭卫公路</td>
      <td>无</td>
      <td>[比瑞吉, 狗粮,  , 小型, 成犬粮, 通用型, 成犬, 狗粮, 2kg,  , 雪纳瑞...</td>
      <td>无</td>
      <td>比瑞吉</td>
    </tr>
    <tr>
      <th>156</th>
      <td>38442639432</td>
      <td>狗粮 比瑞吉泰迪贵宾成犬粮 泰迪狗粮成犬天然粮 比瑞吉狗粮2kg</td>
      <td>贵宾/泰迪(小型犬)成犬专用粮 2KG</td>
      <td>138.0</td>
      <td>337524</td>
      <td>8018</td>
      <td>56434</td>
      <td>69</td>
      <td>34370</td>
      <td>其他</td>
      <td>...</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>比瑞吉宠物用品股份有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海市金山区亭卫公路</td>
      <td>无</td>
      <td>[狗粮,  , 比瑞吉, 泰迪, 贵宾, 成犬粮,  , 泰迪, 狗粮, 成犬, 天然粮, ...</td>
      <td>无</td>
      <td>比瑞吉</td>
    </tr>
    <tr>
      <th>157</th>
      <td>38453114132</td>
      <td>比瑞吉狗粮 小型成犬粮泰迪贵宾比熊博美雪纳瑞通用天然狗粮1.5kg</td>
      <td>(小型犬)成犬海藻胡萝卜粒配方犬粮 1.5KG</td>
      <td>90.0</td>
      <td>81282</td>
      <td>1255</td>
      <td>16747</td>
      <td>45</td>
      <td>12084</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>上海比瑞吉宠物用品股份有限公司</td>
      <td>1500</td>
      <td>中国</td>
      <td>上海市金山区</td>
      <td>海藻胡萝卜粒配方</td>
      <td>[比瑞吉, 狗粮,  , 小型, 成犬粮, 泰迪, 贵宾, 比熊博美, 雪纳瑞, 通用, 天...</td>
      <td>营养均衡配方</td>
      <td>比瑞吉</td>
    </tr>
    <tr>
      <th>158</th>
      <td>38463401287</td>
      <td>狗粮幼犬 比瑞吉小型幼犬粮 泰迪比熊通用 室内小型全价幼犬粮2kg</td>
      <td>(小型犬)幼犬犬粮 2KG</td>
      <td>118.0</td>
      <td>39075</td>
      <td>1852</td>
      <td>9378</td>
      <td>59</td>
      <td>6548</td>
      <td>其他</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>比瑞吉宠物用品股份有限公司</td>
      <td>2000</td>
      <td>中国</td>
      <td>上海市金山区亭卫公路</td>
      <td>无</td>
      <td>[狗粮, 幼犬,  , 比瑞吉, 小型, 幼犬粮,  , 泰迪, 比熊, 通用,  , 室内...</td>
      <td>无</td>
      <td>比瑞吉</td>
    </tr>
    <tr>
      <th>159</th>
      <td>564032233487</td>
      <td>海洋之星SUPERIOR进口营养加强体重控制三文鱼狗粮泰迪通用型12kg</td>
      <td>全犬期精选海洋营养加强体重控制三文鱼配方犬粮 12000g</td>
      <td>999.0</td>
      <td>27</td>
      <td>12</td>
      <td>9</td>
      <td>499</td>
      <td>30</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>12000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>精选海洋营养加强体重控制三文鱼配方</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 体重, 控制, 三文鱼, 狗粮, ...</td>
      <td>三文鱼味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>160</th>
      <td>564179827386</td>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼成犬狗粮泰迪通用型12kg</td>
      <td>成犬精选海洋营养加强三文鱼成犬配方犬粮 12000g</td>
      <td>930.0</td>
      <td>60</td>
      <td>23</td>
      <td>22</td>
      <td>465</td>
      <td>51</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>12000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>精选海洋营养加强三文鱼成犬配方</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, ...</td>
      <td>三文鱼味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>161</th>
      <td>550578330260</td>
      <td>海洋之星SUPERIOR进口营养加强体重控制三文鱼狗粮中大型犬12kg</td>
      <td>成犬Superior（优级）三文鱼控制体重配方大颗粒12kg犬粮 12000g</td>
      <td>999.0</td>
      <td>77</td>
      <td>26</td>
      <td>35</td>
      <td>499</td>
      <td>35</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>12000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>Superior（优级）三文鱼控制体重配方大颗粒12kg</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 体重, 控制, 三文鱼, 狗粮, ...</td>
      <td>三文鱼味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>162</th>
      <td>550548533749</td>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼成犬狗粮中大型犬12kg</td>
      <td>成犬鱼肉味犬粮 12000g</td>
      <td>930.0</td>
      <td>104</td>
      <td>21</td>
      <td>29</td>
      <td>465</td>
      <td>37</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>12000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>鱼肉味</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, ...</td>
      <td>鱼肉口味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>163</th>
      <td>564102078063</td>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼幼犬狗粮中大型犬12kg</td>
      <td>(中大型犬)幼犬精选海洋营养加强三文鱼幼犬配方犬粮 12000g</td>
      <td>970.0</td>
      <td>20</td>
      <td>9</td>
      <td>4</td>
      <td>485</td>
      <td>34</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>12000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>精选海洋营养加强三文鱼幼犬配方</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 幼犬, 狗粮, ...</td>
      <td>三文鱼味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>164</th>
      <td>543492643164</td>
      <td>海洋之星狗粮金毛阿拉斯加大型犬幼犬无谷天然粮大颗粒犬粮12kg</td>
      <td>(中大型犬)幼犬犬粮 12000g</td>
      <td>750.0</td>
      <td>1936</td>
      <td>100</td>
      <td>539</td>
      <td>375</td>
      <td>689</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>鱼肉味</td>
      <td>[海洋之星, 狗粮, 金毛, 阿拉斯加, 大型犬, 幼犬, 无谷, 天然粮, 大颗粒, 犬粮...</td>
      <td>鱼肉口味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>165</th>
      <td>550575110336</td>
      <td>海洋之星SUPERIOR原装进口营养加强体重控制三文鱼配方狗粮6kg</td>
      <td>成犬Superior（优级）三文鱼控制体重小颗粒6kg犬粮 6000g</td>
      <td>570.0</td>
      <td>140</td>
      <td>48</td>
      <td>52</td>
      <td>285</td>
      <td>66</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>6000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>Superior（优级）三文鱼控制体重小颗粒6kg</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 体重, 控制, 三文鱼, ...</td>
      <td>三文鱼味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>166</th>
      <td>548685638984</td>
      <td>海洋之星SUPERIOR进口营养加强体重控制三文鱼狗粮通用型1.5kg</td>
      <td>成犬Superior（优级）体重控制小颗粒1.5kg犬粮 1500g</td>
      <td>230.0</td>
      <td>496</td>
      <td>58</td>
      <td>166</td>
      <td>115</td>
      <td>65</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>1500</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>Superior（优级）体重控制小颗粒1.5kg</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 体重, 控制, 三文鱼, 狗粮, ...</td>
      <td>鱼肉口味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>167</th>
      <td>550550421444</td>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼幼犬狗粮泰迪通用型6kg</td>
      <td>幼犬三文鱼犬粮 6000g</td>
      <td>550.0</td>
      <td>175</td>
      <td>29</td>
      <td>55</td>
      <td>275</td>
      <td>56</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>6000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>三文鱼</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 幼犬, 狗粮, ...</td>
      <td>三文鱼味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>168</th>
      <td>550572214265</td>
      <td>海洋之星SUPERIOR进口营养加强三文鱼成犬狗粮泰迪通用型1.5kg</td>
      <td>(中小型犬)成犬三文鱼犬粮 1500g</td>
      <td>190.0</td>
      <td>1048</td>
      <td>203</td>
      <td>337</td>
      <td>95</td>
      <td>135</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>1500</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>三文鱼</td>
      <td>[海洋之星, SUPERIOR, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, 泰迪, ...</td>
      <td>三文鱼味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>169</th>
      <td>534010155073</td>
      <td>大幼试吃装中大型犬幼犬金毛阿拉斯牧羊犬哈士奇狗粮三包*30g</td>
      <td>(大型犬)幼犬深海鱼配方犬粮 100g</td>
      <td>9.9</td>
      <td>7543</td>
      <td>206</td>
      <td>1836</td>
      <td>4</td>
      <td>476</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>海洋之星</td>
      <td>100</td>
      <td>其他</td>
      <td>Fish4Dogs.Ltd</td>
      <td>深海鱼配方</td>
      <td>[大幼, 试吃装, 中大型犬, 幼犬, 金毛, 阿拉斯, 牧羊犬, 哈士奇, 狗粮, 三包,...</td>
      <td>深海鱼味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>170</th>
      <td>550574250453</td>
      <td>海洋之星SUPERIOR原装进口营养加强三文鱼成犬狗粮泰迪通用型6kg</td>
      <td>成犬鱼肉犬粮 6000g</td>
      <td>520.0</td>
      <td>278</td>
      <td>85</td>
      <td>76</td>
      <td>260</td>
      <td>85</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>United Petfood Producers NV</td>
      <td>6000</td>
      <td>比利时</td>
      <td>Terdonkkaai 16, 9042 Gent, Belgium</td>
      <td>鱼肉</td>
      <td>[海洋之星, SUPERIOR, 原装, 进口, 营养, 加强, 三文鱼, 成犬, 狗粮, ...</td>
      <td>鱼肉口味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>171</th>
      <td>534139840517</td>
      <td>试吃装 深海鱼三文鱼成犬中大型犬试吃装大颗粒阿拉斯加三包*30g</td>
      <td>(大型犬)成犬鱼肉味犬粮 100g</td>
      <td>9.9</td>
      <td>7029</td>
      <td>238</td>
      <td>1519</td>
      <td>4</td>
      <td>247</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>Fish4Dogs.Ltd</td>
      <td>100</td>
      <td>其他</td>
      <td>Fish4Dogs.Ltd</td>
      <td>鱼肉味</td>
      <td>[试吃装,  , 深海鱼, 三文鱼, 成犬, 中大型犬, 试吃装, 大颗粒, 阿拉斯加, 三...</td>
      <td>鱼肉口味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>172</th>
      <td>529421904631</td>
      <td>海洋之星深海鱼狗粮无谷天然粮中小型犬成犬主粮小颗粒6kg</td>
      <td>成犬无谷物深海鱼小颗粒配方犬粮 其它</td>
      <td>410.0</td>
      <td>5723</td>
      <td>275</td>
      <td>2018</td>
      <td>205</td>
      <td>717</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>6000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物深海鱼小颗粒配方</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 中小型犬, 成, 犬主粮, 小颗粒, 6kg]</td>
      <td>无谷深海鱼配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>173</th>
      <td>521099649915</td>
      <td>海洋之星深海鱼狗粮无谷天然粮金毛萨摩耶大型犬成犬大颗粒12kg</td>
      <td>成犬无谷物深海鱼大颗粒配方犬粮 12kg</td>
      <td>710.0</td>
      <td>6287</td>
      <td>311</td>
      <td>1950</td>
      <td>355</td>
      <td>929</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物深海鱼大颗粒配方</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 金毛, 萨摩耶, 大型犬, 成犬, 大颗粒...</td>
      <td>无谷深海鱼配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>174</th>
      <td>521103914337</td>
      <td>海洋之星深海鱼狗粮无谷天然粮中小型犬幼犬小颗粒狗粮6kg</td>
      <td>幼犬无谷物深海鱼小颗粒6kg犬粮</td>
      <td>440.0</td>
      <td>7869</td>
      <td>245</td>
      <td>2504</td>
      <td>220</td>
      <td>1578</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>6000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>幼犬无谷物深海鱼小颗粒6kg</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 中小型犬, 幼犬, 小颗粒, 狗粮, 6kg]</td>
      <td>无谷深海鱼配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>175</th>
      <td>521091943822</td>
      <td>海洋之星深海鱼狗粮无谷天然粮中小型犬幼犬小颗粒12kg</td>
      <td>幼犬无谷物深海鱼幼犬配方小颗粒犬粮 12kg</td>
      <td>750.0</td>
      <td>7739</td>
      <td>264</td>
      <td>2230</td>
      <td>375</td>
      <td>2027</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物深海鱼幼犬配方小颗粒</td>
      <td>[海洋之星, 深海鱼, 狗粮, 无谷, 天然粮, 中小型犬, 幼犬, 小颗粒, 12kg]</td>
      <td>无谷深海鱼配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>176</th>
      <td>521092399322</td>
      <td>海洋之星狗粮三文鱼无谷天然粮金毛拉布拉多大型犬成犬主粮 12kg</td>
      <td>成犬无谷物三文鱼大颗粒配方犬粮 12kg</td>
      <td>710.0</td>
      <td>10751</td>
      <td>452</td>
      <td>2668</td>
      <td>355</td>
      <td>2130</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>12000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物三文鱼大颗粒配方</td>
      <td>[海洋之星, 狗粮, 三文鱼, 无谷, 天然粮, 金毛, 拉布拉多, 大型犬, 成, 犬主粮...</td>
      <td>无谷三文鱼配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>177</th>
      <td>521323201615</td>
      <td>海洋之星深海鱼成犬狗粮无谷天然粮中小型犬成犬小颗粒1.5kg</td>
      <td>成犬无谷物深海鱼小颗粒1.5kg犬粮</td>
      <td>150.0</td>
      <td>10626</td>
      <td>352</td>
      <td>3402</td>
      <td>75</td>
      <td>1338</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>1500</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>成犬无谷物深海鱼小颗粒1.5kg</td>
      <td>[海洋之星, 深海鱼, 成犬, 狗粮, 无谷, 天然粮, 中小型犬, 成犬, 小颗粒, 1....</td>
      <td>无谷深海鱼配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>178</th>
      <td>521103456733</td>
      <td>海洋之星深海鱼成犬狗粮无谷天然粮中小型犬成犬狗粮12kg小颗粒</td>
      <td>成犬无谷物深海鱼大颗粒配方犬粮 其它</td>
      <td>710.0</td>
      <td>7173</td>
      <td>388</td>
      <td>2226</td>
      <td>355</td>
      <td>1171</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>6000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物深海鱼大颗粒配方</td>
      <td>[海洋之星, 深海鱼, 成犬, 狗粮, 无谷, 天然粮, 中小型犬, 成犬, 狗粮, 12k...</td>
      <td>无谷深海鱼配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>179</th>
      <td>545052402563</td>
      <td>海洋之星狗粮试吃装泰迪比熊玩具犬幼犬成犬全犬期小颗粒三包*30g</td>
      <td>全犬期鱼味犬粮 100g</td>
      <td>9.9</td>
      <td>10623</td>
      <td>349</td>
      <td>3003</td>
      <td>12</td>
      <td>578</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>海洋之星</td>
      <td>100</td>
      <td>其他</td>
      <td>Fish4Dogs.Ltd</td>
      <td>鱼味</td>
      <td>[海洋之星, 狗粮, 试吃装, 泰迪, 比熊, 玩具犬, 幼犬, 成犬, 全犬期, 小颗粒,...</td>
      <td>鱼肉口味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>180</th>
      <td>521103920550</td>
      <td>海洋之星狗粮通用型天然粮中小型犬幼犬主粮泰迪博美比熊1.5kg</td>
      <td>幼犬无谷物深海鱼幼犬配方大颗粒犬粮 1.5kg</td>
      <td>160.0</td>
      <td>25402</td>
      <td>614</td>
      <td>7582</td>
      <td>80</td>
      <td>7682</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>山东海创工贸有限公司</td>
      <td>1500</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物深海鱼幼犬配方大颗粒</td>
      <td>[海洋之星, 狗粮, 通用型, 天然粮, 中小型犬, 幼犬主粮, 泰迪, 博美比, 熊, 1...</td>
      <td>无谷深海鱼配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>181</th>
      <td>534102297282</td>
      <td>海洋之星狗粮试吃装泰迪比熊贵宾幼犬通用型小颗粒美毛三包*30g</td>
      <td>(中小型犬)幼犬深海鱼配方犬粮 100g</td>
      <td>9.9</td>
      <td>30387</td>
      <td>1018</td>
      <td>8754</td>
      <td>4</td>
      <td>2003</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>犬粮</td>
      <td>通用型</td>
      <td>海洋之星</td>
      <td>100</td>
      <td>其他</td>
      <td>Fish4Dogs.Ltd</td>
      <td>深海鱼配方</td>
      <td>[海洋之星, 狗粮, 试吃装, 泰迪, 比熊, 贵宾, 幼犬, 通用型, 小颗粒, 美毛, ...</td>
      <td>深海鱼味配方</td>
      <td>海洋之星</td>
    </tr>
    <tr>
      <th>182</th>
      <td>521100097514</td>
      <td>海洋之星三文鱼狗粮泰迪吉娃娃贵宾玩具犬狗粮无谷天然粮6kg</td>
      <td>贵宾/泰迪无谷物三文鱼玩具犬配方犬粮</td>
      <td>440.0</td>
      <td>8286</td>
      <td>573</td>
      <td>2659</td>
      <td>220</td>
      <td>1750</td>
      <td>鱼肉味</td>
      <td>...</td>
      <td>专用粮</td>
      <td>贵宾/泰迪</td>
      <td>山东海创工贸有限公司</td>
      <td>6000</td>
      <td>中国</td>
      <td>山东省聊城市经济开发区牡丹江路8号</td>
      <td>无谷物三文鱼玩具犬配方</td>
      <td>[海洋之星, 三文鱼, 狗粮, 泰迪, 吉娃娃, 贵宾, 玩具犬, 狗粮, 无谷, 天然粮,...</td>
      <td>无谷三文鱼配方</td>
      <td>海洋之星</td>
    </tr>
  </tbody>
</table>
<p>183 rows × 23 columns</p>

</div>

