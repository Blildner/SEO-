Python数据分析常用高阶函数大全
==================================

[TOC]



map
=====

`map(function,iterable,...)`

第一个参数，是函数

第二个参数，是可迭代对象（列表、字符串等）

map返回的是对`可迭代对象`里的每个元素进行`函数`运算的结果

例如：

```
def fun(x):
    return x*3
```

```
l=[0,1,2,3,4,5]
l_m=map(fun,l)
print(list(l_m))
```

原本是

`[0,1,2,3,4,5]`

运行map后返回的结果是

`[0, 3, 6, 9, 12, 15]`



相当于对`可迭代对象`里的每个元素都进行了*3的运算，也就是我们给定`函数`运算的方式，然后返回一个值。

这里需要注意的是 ，map()直接返回的是一个`<map at 0x205ef31fb00>`的对象

我们需要利用list函数将它里边的元素释放出来。

![1566023536625](Python数据分析常用高阶函数大全.assets/1566023536625.png)

与此同时，map函数的好朋友就是lambda，lambda匿名函数经常作为map的第一个参数进行组合使用

例如

```
print(list(map(lambda x:x*3,l)))
```

返回的结果依旧是

`[0, 3, 6, 9, 12, 15]`





zip
=====



zip()将多个可迭代对象的元素组合成为为一个元组序列

```
l  = ['a', 'b', 'c']
n = [1, 2, 3]
print(list(zip(l,n)))
```



`[('a', 1), ('b', 2), ('c', 3)]`

和map类似，zip返回的也是一个zip的元组迭代器对象，我们需要使用list将它的元素释放出来



filter
======

*filter（function，sequence）*

第一个参数是函数，第二个参数是可迭代对象

最后返回的是，可迭代对象里满足函数要求的元素。

因此也称之为过滤。

```python
long = [1,2,3,4,5]
list(filter(lambda x:x%2==0,long)) # 找出偶数。
# filter函数返回的是迭代器，所以需要用list转换,进行释放元素。
# 输出：
[2, 4]
```



reduce
======

 *reduce（function，iterable）*

第一个参数是函数，第二个参数是可迭代对象（列表，字符串等）

导入reduce的时候需要用到funtools这模块

```python
from functools import reduce 

lk = [2,3,4]
reduce(lambda y,z:z+y,lk)
# out : 9
```

运算的步骤是

2+3=5

5+4=9

最后返回的结果就是9



apply
=====

*DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)*

apply函数是pandas.DataFrame里的方法

例如

kk是pd.DataFrame的类型的数据

```
    0
0  0a
1  1b
2  2c
3  3d
4  4e
```


```python
kk["new"]=kk[0].apply(lambda x:x[-1] )
kk
```
```
  0  new
0 0a   a
1 1b   b
2 2c   c
3 3d   d
4 4e   e

```



sort_values
===========

## 参数    

*DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')*

参数说明    
------------

```python
axis:{0 or ‘index’, 1 or ‘columns’}#, default 0，默认按照索引排序，即纵向排序，如果为1，则是横向排序    
by:str or list of str#；如果axis=0，那么by="列名"；如果axis=1，那么by="行名"；  
ascending:#布尔型，True则升序，可以是[True,False]，即第一字段升序，第二个降序  
inplace:#布尔型，是否用排序后的数据框替换现有的数据框  
kind:排序方法，#{‘quicksort’, ‘mergesort’, ‘heapsort’}, default ‘quicksort’。
na_position : #{‘first’, ‘last’}, default ‘last’，默认缺失值排在最后面
```

```python
import pandas as pd
import numpy as np
a = np.random.randint(low=0,high=100,size=(11,2))
data = pd.DataFrame(a)
data.apply(lambda x:x*10)

[*data.columns]=["z1",'z2']

```

|      | z1   | z2   |
| ---: | ---- | ---- |
|    0 | 16   | 13   |
|    1 | 57   | 0    |
|    2 | 36   | 16   |
|    3 | 76   | 86   |
|    4 | 88   | 64   |
|    5 | 12   | 24   |
|    6 | 86   | 59   |
|    7 | 28   | 61   |
|    8 | 44   | 29   |
|    9 | 56   | 91   |
|   10 | 5    | 4    |

```python
data.sort_values(by="z1",ascending= False) 
```

|      | z1   | z2   |
| ---: | ---- | ---- |
|    4 | 88   | 64   |
|    6 | 86   | 59   |
|    3 | 76   | 86   |
|    1 | 57   | 0    |
|    9 | 56   | 91   |
|    8 | 44   | 29   |
|    2 | 36   | 16   |
|    7 | 28   | 61   |
|    0 | 16   | 13   |
|    5 | 12   | 24   |
|   10 | 5    | 4    |

```python
data.sort_values(by="z2",ascending= False)
```

|      | z1   | z2   |
| ---: | ---- | ---- |
|    9 | 56   | 91   |
|    3 | 76   | 86   |
|    4 | 88   | 64   |
|    7 | 28   | 61   |
|    6 | 86   | 59   |
|    8 | 44   | 29   |
|    5 | 12   | 24   |
|    2 | 36   | 16   |
|    0 | 16   | 13   |
|   10 | 5    | 4    |
|    1 | 57   | 0    |



```python
import random
random.seed=1234
import pandas as pd
import numpy as np
#a=np.random.randint(low=0,high=100,size=(10,6))
data = pd.DataFrame(a)
data.apply(lambda x:x*10)

[*data.columns]=["z1",'z2',"z3",'z4',"z5",'z6']
data.sort_values(by=8,ascending= False,axis=1) 
```

|       | z3     | z4     | z1     | z2     | z5     | z6    |
| ----: | ------ | ------ | ------ | ------ | ------ | ----- |
|     0 | 89     | 63     | 65     | 45     | 61     | 84    |
|     1 | 51     | 18     | 75     | 22     | 28     | 29    |
|     2 | 44     | 64     | 18     | 13     | 51     | 81    |
|     3 | 18     | 29     | 17     | 47     | 4      | 53    |
|     4 | 93     | 85     | 15     | 83     | 29     | 70    |
|     5 | 19     | 74     | 33     | 83     | 15     | 45    |
|     6 | 76     | 66     | 53     | 21     | 35     | 48    |
|     7 | 58     | 46     | 31     | 40     | 93     | 55    |
| **8** | **95** | **93** | **87** | **54** | **11** | **7** |
|     9 | 93     | 62     | 17     | 42     | 65     | 80    |



sort
====

*sort(key,reverse)*

这个是列表的方法

key：是排序的条件

reverse：表示是否逆序，默认是从小到大，默认为False

```python
x = ['mmm', 'mm', 'mm', 'm' ]
x.sort(key = len)
print (x)
# out： ['m', 'mm', 'mm', 'mmm']


y = [3, 2, 8 ,0 , 1]
y.sort(reverse = True)
print (y) 
#[8, 3, 2, 1, 0]
#True为逆序排列，False为正序排列
```



sorted
======

对所有可迭代对象都可以排序。

而且不会改变原有的可迭代对象的结构，而是生成一个新的数据。

```python
 #sorted(L)返回一个排序后的L，不改变原始的L
L=[('b',2),('a',100),('c',30),('d',48)]
sorted(L, key=lambda x:x[1])
# out：
# [('b', 2), ('c', 30), ('d', 48), ('a', 100)]
sorted(L, key=lambda x:x[0])
# out：[('a', 100), ('b', 2), ('c', 30), ('d', 48)]
```

Enumerate
=========

enumerate 是一个会返回元组迭代器的内置函数，这些元组包含列表的索引和值。当你需要在循环中获取可迭代对象的每个元素及其索引时，将经常用到该函数。

示例代码:

```
letters = ['a', 'b', 'c', 'd', 'e']
for i, letter in enumerate(letters):
  print(i, letter)
```

返回的结果

```
0 a
1 b
2 c
3 d
4 e
```



练习题
------

**Python 中的 Zip 和 Enumerate[相关练习]**

使用 zip 写一个 for 循环，该循环会创建一个字符串，指定每个点的标签和坐标，并将其附加到列表 points。每个字符串的格式应该为 label: x, y, z。例如，第一个坐标的字符串应该为 F: 23, 677, 4。

**参考答案：**

```
x_coord = [23, 53, 2, -12, 95, 103, 14, -5]
y_coord = [677, 233, 405, 433, 905, 376, 432, 445]
z_coord = [4, 16, -6, -42, 3, -6, 23, -1]
labels = ["F", "J", "A", "Q", "Y", "B", "W", "X"]
points = []
# write your for loop here
for label, x, y, z in zip(labels, x_coord, y_coord, z_coord):
  points.append(label+": " + str(x) + ', ' + str(y) + ', ' + str(z))
for point in points:
  print(point)
```

输出如下：

> F: 23, 677, 4
> J: 53, 233, 16
> A: 2, 405, -6
> Q: -12, 433, -42
> Y: 95, 905, 3
> B: 103, 376, -6
> W: 14, 432, 23
> X: -5, 445, -1

使用 zip 创建一个字段 cast，该字典使用 names 作为键，并使用 heights 作为值。

**参考答案：**

```
cast_names = ["Barney", "Robin", "Ted", "Lily", "Marshall"]
cast_heights = [72, 68, 72, 66, 76]
cast = dict(zip(cast_names,cast_heights))
print(cast)
```

输出：

> {'Barney': 72, 'Ted': 72, 'Robin': 68, 'Lily': 66, 'Marshall': 76}

将 cast 元组拆封成两个 names 和 heights 元组。

**参考答案：**

```
cast = (("Barney", 72), ("Robin", 68), ("Ted", 72), ("Lily", 66), ("Marshall", 76))
# define names and heights here
names,heights = zip(*cast)
print(names) # ('Barney', 'Robin', 'Ted', 'Lily', 'Marshall')
print(heights) # (72, 68, 72, 66, 76)
```

使用 zip 将 data 从 4x3 矩阵转置成 3x4 矩阵。

**参考答案：**

```
data = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
data_transpose = tuple(zip(*data))
print(data_transpose) # ((0, 3, 6, 9), (1, 4, 7, 10), (2, 5, 8, 11))
```

使用 enumerate 修改列表 cast，使每个元素都包含姓名，然后是角色的对应身高。例如，cast 的第一个元素应该从 “Barney Stinson” 更改为 "Barney Stinson 72”。

**参考答案：**

```
cast = ["Barney Stinson", "Robin Scherbatsky", "Ted Mosby", "Lily Aldrin", "Marshall Eriksen"]
heights = [72, 68, 72, 66, 76]
for i, c in enumerate(cast):
  cast[i] += ' ' + str(heights[i])
print(cast) # ['Barney Stinson 72', 'Robin Scherb
```



推导式
======

推导式comprehensions（又称解析式），是Python的一种独有特性。推导式是可以从一个数据序列构建另一个新的数据序列的结构体。 共有三种推导，在Python2和3中都有支持：

- 列表(list)推导式
- 字典(dict)推导式
- 集合(set)推导式

 

列表推导式
----------

**1、使用[]生成list**

例一：

```
multiples = [i for i in range(20) if i % 5 is 0]
print(multiples)
# Output:[0, 5, 10, 15]
```

 

例二：

```
def sd(x):
    return x*x

multiples = [sd(i) for i in range(20) if i % 5 is 0]
print (multiples)
#  Output: [0, 25, 100, 225]
```

  

 

字典推导式
----------

字典推导和列表推导的使用方法是类似的，只不过中括号该改成大括号。直接举例说明：

```
m = {'a': 200, 'b': 56}
ma = {v: k for k, v in m.items()}
print (ma)
#  Output: {200: 'a', 56: 'b'}
```

 

 

集合推导式
----------

它们跟列表推导式也是类似的。 唯一的区别在于它使用大括号{}。

**例一：**

```
squared = {x**2 for x in [1, 1, 2, 2]}
print(squared)
# Output: set([1, 4])
```

集合推导式有一个好处就是可以做到去重









collections模块的Counter类
==========================

Python标准库——collections模块的Counter类

collections模块包含了dict、set、list、tuple以外的一些特殊的容器类型，分别是：

- OrderedDict类：排序字典，是字典的子类。
- namedtuple()函数：命名元组，是一个工厂函数。
- Counter类：为hashable对象计数，是字典的子类。
- deque：双向队列。
- defaultdict：使用工厂函数创建字典，使不用考虑缺失的字典键。

Counter类
---------

Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key，其计数作为value。计数值可以是任意的Interger（包括0和负数）。Counter类和其他语言的bags或multisets很相似。

### 创建

下面的代码说明了Counter类创建的四种方法：

Counter类的创建

Python

c = Counter()  # 创建一个空的Counter类<br />c = Counter('gallahad')  # 从一个可iterable对象（list、tuple、dict、字符串等）创建><br />c = Counter({'a': 4, 'b': 2})  # 从一个字典对象创建 c = Counter(a=4, b=2)  # 从一组键值对创建  

### 计数值的访问与缺失的键

当所访问的键不存在时，返回0，而不是KeyError；否则返回它的计数。

计数值的访问

 c = Counter("abcdefgab")

 c["a"]    2

 c["c"]    1

 c["h"]    0



### 计数器的更新（update和subtract）

可以使用一个iterable对象或者另一个Counter对象来更新键值。

计数器的更新包括增加和减少两种。其中，增加使用update()方法：

计数器的更新（update）

c = Counter('which')

c.update('witch')  # 使用另一个iterable对象更新

c['h']3

d = Counter('watch')

c.update(d)  # 使用另一个Counter对象更新

c['h']  4



减少则使用subtract()方法：

计数器的更新（subtract）

Python

c = Counter('which')

c.subtract('witch')  # 使用另一个iterable对象更新

c['h']1

d = Counter('watch')

c.subtract(d)  # 使用另一个Counter对象更新

c['a']-1

### 键的删除

当计数值为0时，并不意味着元素被删除，删除元素应当使用`del`。

键的删除

Python

c = Counter("abcdcba")

c=Counter({'a': 2, 'c': 2, 'b': 2, 'd': 1})

c["b"] = 0

c=Counter({'a': 2, 'c': 2, 'd': 1, 'b': 0})

del c["a"]

c=Counter({'c': 2, 'b': 2, 'd': 1})

###  elements()

返回一个迭代器。元素被重复了多少次，在该迭代器中就包含多少个该元素。元素排列无确定顺序，个数小于1的元素不被包含。

elements()方法

 c = Counter(a=4, b=2, c=0, d=-2)

 list(c.elements())['a', 'a', 'a', 'a', 'b', 'b']

###  most_common([n])

返回一个TopN列表。如果n没有被指定，则返回所有元素。当多个元素计数值相同时，排列是无确定顺序的。

most_common()方法

Python

c = Counter('abracadabra')

c.most_common()[('a', 5), ('r', 2), ('b', 2), ('c', 1), ('d', 1)]

c.most_common(3)[('a', 5), ('r', 2), ('b', 2)]



### 浅拷贝copy

浅拷贝copy

Python

 c = Counter("abcdcba")

cCounter({'a': 2, 'c': 2, 'b': 2, 'd': 1})

d = c.copy()>>> dCounter({'a': 2, 'c': 2, 'b': 2, 'd': 1})

### 算术和集合操作

+、-、&、|操作也可以用于Counter。其中&和|操作分别返回两个Counter对象各元素的最小值和最大值。需要注意的是，得到的Counter对象将删除小于1的元素。

Counter对象的算术和集合操作

Python

c = Counter(a=3, b=1)

 d = Counter(a=1, b=2)

 c + d  # c[x] + d[x]Counter({'a': 4, 'b': 3})

 c - d  # subtract（只保留正数计数的元素）Counter({'a': 2}

 c & d  # 交集:  min(c[x], d[x])Counter({'a': 1, 'b': 1})

 c \| d  # 并集:  max(c[x], d[x])Counter({'a': 3, 'b': 2})



### 常用操作

下面是一些Counter类的常用操作，来源于Python官方文档

Counter类常用操作

sum(c.values())  # 所有计数的总数

c.clear()  # 重置Counter对象，注意不是删除

list(c) # 将c中的键转为列表

set(c)  #将c中的键转为set 

dict(c)  # 将c中的键值对转为字典

c.items()  # 转为(elem, cnt)格式的列表

Counter(dict(list_of_pairs))  # 从(elem, cnt)格式的列表转换为Counter类对象

c.most_common()[:-n:-1]  # 取出计数最少的n-1个元素

c += Counter()  # 移除0和负值