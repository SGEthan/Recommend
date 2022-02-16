# 推荐系统实验说明

## 运行环境

本次实验我们没有使用特殊的包，仅使用了一些科学计算工具，具体环境为

* Windows 11
* conda 4.10.3

## 关键函数说明

本次实验我们调用了以下工具

```python
import numpy as np
import os
import scipy.sparse as ss
import gc
from sklearn.preprocessing import normalize
```

几个关键函数的定义如下：

* `data_loader()`：用于读入原始数据集并进行处理，以稀疏矩阵的形式将数据集存储于内存并导出到硬盘中
* `normalized(raw_matrix)`：对原始矩阵进行规范处理，具体处理方式见实验报告
* `compute_item_sim(normalized_matrix)`：计算不同物品之间的相似度
* `compute_pred()`：计算每个用户对于不同物品的预测评分
* `predict()`：对于上一步得到的评分进行排序，并给出top100结果

## 编译运行方式

本项目仅含一个`python `脚本，直接运行`python main.py`即可

