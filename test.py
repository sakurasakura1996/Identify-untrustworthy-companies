import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
print(df.shape)
print(type(df))
print(type(df['行业代码'][3]))
print(df['企业类型'].max())
ass = df['企业类型'].value_counts()
print(ass)
# ass.plot.hist(grid=True,rwidth=0.5)


df['企业类型'].plot.hist(grid=True, rwidth=0.5,)
df['登记机关'].plot.hist(grid=True, rwidth=0.5)
plt.show()

# #
# #
# # str = "123456"
# # print(str[2:4])
# df = pd.DataFrame(columns=['data'],index=[1,2,3,4],data=[['234'],
#                                                          ['234'],
#                                                          ['234'],
#                                                          ['234']])
# df['data'] = df['data'].str[1:2]
# print(type(df['data'][2]))


# params = {
#     'bagging_freq': 1,
#     'bagging_fraction': 0.85,
#     'bagging_seed': int(np.random.rand() * 100),
#     'boost': 'gbdt',
#     'feature_fraction': 0.85,     # 随机选择85%的特征
#     'feature_fraction_seed': int(np.random.rand() * 100),
#     'learning_rate': 0.01,
#     'max_depth': -1,     # 貌似 8 改成10 提高了
#     'metric': 'auc',
#     'min_data_in_leaf': 20,    # 一个叶子上数据的最小数量. 可以用来处理过拟合. default 20
#     'num_leaves': 1024,
#     'num_threads': 4,
#     'objective': 'binary',
#     "lambda_l1": 0.5,
#     'lambda_l2': 1.2,
#     'verbosity': 1,
#     'is_unbalance': True
# }
# best_score = 0.0
# best_max_depth = 1
# best_num_leaves = 1
# for i in range(10, 15):
#     max_dep = 2 ** (i-1)
#     min_dep = 2 ** (i - 4)
#
#     step_stride = int((max_dep - min_dep) / 100)
#     num_leave = []
#     for j in range(1, 101):
#         num = min_dep + step_stride * j
#         num_leave.append(num)
#     for j in num_leave:
#         params['max_depth'] = i
#         params['num_leaves'] = j
#         print(params)