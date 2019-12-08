
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import warnings
from jupyterthemes import jtplot


jtplot.style()
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


class BaseModel:
    def __init__(self):
        self.train = None  # 训练集
        self.test = None  # 测试集
        self.train_label = None  # 训练集标签
        self.features = None  # 训练时使用的参数
        self.cat_feats = None  # 训练时使用的参数中的类别参数

    def load_data(self, path):
        """
        读取数据
        @param: path:数据存放根目录
        """
        print('Loading data...', end='\t')
        self.train = pd.read_csv('train.csv')
        self.test = pd.read_csv('test.csv')
        self.train_label = pd.read_csv('train_label.csv')
        self.submit = pd.read_csv('submission.csv')
        print("train data shape:"+str(self.train.shape))
        print('Done!')

    def _missing_counter(self, df, cols=None):
        """
        统计数据缺失情况
        @param: df:数据集
        @param: cols:要统计缺失情况的特征列表
        """
        if cols is None:
            cols = df.columns
        counter = pd.DataFrame(columns=['Feature', 'Count', 'Percent'])
        length = df.shape[0]
        counter['Feature'] = cols
        counter = counter.set_index('Feature')
        for f in cols:
            n = sum(df[f].isna())
            p = n / length
            counter['Count'].loc[f] = n
            counter['Percent'].loc[f] = p
        return counter[counter['Count'] != 0]

    def preprocess(self):
        """
        数据预处理（数据清洗等）
        """
        print('Preprocessing...', end='\t')
        # 处理缺失值
        counter = self._missing_counter(self.train)
        counter.reset_index(inplace=True)
        feats1 = counter[counter['Count'] == 8343]['Feature'].values.tolist()  # 缺失值达到8343个的特征
        counter = counter[counter['Count'] != 8343]
        counter = counter[counter['Percent'] < 0.5]
        feats2 = counter['Feature'].values.tolist()            # 缺失值在一半以下的特征
        feats = feats1 + feats2
        all_df = pd.concat([self.train, self.test])
        for df in [self.train, self.test]:
            df = df[feats]
            for f in feats2:
                if df[f].dtype != object:
                    df[f] = df[f].fillna(all_df[f][~all_df[f].isna()].mean())
        print("Done!")

    def feature_engineering(self):
        """
        特征工程
        """
        print('Feature engineering ...', end='\t')
        self.train = pd.merge(self.train, self.train_label, on='ID', how='left')
        # * ---------------------------------------------------------------------------
        train = self.train.copy()
        test = self.test.copy()
        print(train.shape)
        for i in range(train.shape[1]):
            print(train.columns[i])
        for df in [train, test]:
            df['总税'] = df['印花税'] + df['增值税'] + df['企业所得税'] + df['城建税'] + df['教育费']
            df['注册资本税收比'] = df['注册资本'] / df['总税']
            # 通过查询邮政编码的含义，中间两位代表市，可能提取出来作用更大一些
            df['邮政编码'] = df['邮政编码'].str[2:4]
            # 看到图中的行业代码影响力很大。所以查询以下四位数有什么代表含义。四位数包括 大类两位数，中类三位数（包括大类的两位数），小类四位数
            # 所以想把大类 或者种类单独提取出来，行业代码是float64位的数而不是字符串
            df['行业代码'] = df['行业代码'].astype(str).str[:2]
            # df['长期负债合计_差值'] = df['长期负债合计_年末数'] - df['长期负债合计_年末数']
            # df['长期借款_差值'] = df['长期借款_年末数'] - df['长期借款_年初数']
            # df['货币资金_差值'] = df['货币资金_年末数']- df['货币资金_年初数']
            # df['未分配利润_差值'] = df['未分配利润_年末数'] - df['未分配利润_年初数']
            # df['其他应收款_差值'] = df['其他应收款_年末数'] - df['其他应收款_年初数']
            # df['所有者权益合计_差值'] = df['所有者权益合计_年末数'] - df['所有者权益合计_年初数']
            # df['应收账款_差值'] = df['应收账款_年末数']- df['应收账款_年初数']
            df['企业所得税与增值税之比'] = df['企业所得税']/df['增值税']

            # df['负债_年初'] = df['长期负债合计_年初数'] + df['长期借款_年初数'] + df['长期应付款_年初数'] + df
            # ******************
            # 在这里添加自己的特征工程部分，参考上两行
            # ******************

        self.train = train.copy()
        self.test = test.copy()
        # * ---------------------------------------------------------------------------
        self.features = [_ for _ in self.train.columns if
                         _ not in ['ID', 'Label', '经营范围', '经营期限至', '核准日期', '注销时间', '经营期限自', '成立日期']]
        # 解决新版本LGB输入数据集不支持中文特征的问题：临时将中文特征编码为整数
        map_columns = {self.features[i]: i for i in range(len(self.features))}
        self.train.rename(columns=map_columns, inplace=True)
        self.test.rename(columns=map_columns, inplace=True)
        self.cat_feats = ['企业类型', '登记机关', '企业状态', '邮政编码', '行业代码', '行业门类', '企业类别', '管辖机关']
        self.features = [map_columns[i] for i in self.features]
        self.cat_feats = [map_columns[i] for i in self.cat_feats]
        self.map_columns = {i[1]: i[0] for i in map_columns.items()}
        print(self.train.shape)
        print(len(self.features))
        print(len(self.cat_feats))
        print("Done!")

    def gen_dataset(self, path=''):
        """
        高级封装，数据读取+数据预处理+特征工程
        @param: path:数据存储的根目录
        """
        self.load_data(path)
        self.preprocess()
        self.feature_engineering()
        print("self.train.shape"+str(self.train.shape))

    def model_train(self, model, params, seed, early_stop=200):
        """
        模型训练
        @param: model:模型类型
        @param: params:模型参数
        @param: seed:随机数种子
        @param: early_stop:模型训练时的早停参数
        """
        if model == 'LGB':
            oof, predictions, feature_importance_df = self._lgb_model(params, seed, early_stop)
        return oof, predictions, feature_importance_df

    def _lgb_model(self, params, seed=4545, early_stop=200):
        """
        使用LightGBM进行五折交叉训练
        @param: params:参数
        @param: seed:五折交叉验证时的随机数种子
        @param: early_stop:模型训练时的早停参数
        """
        train = self.train.copy()
        test = self.test.copy()
        target = train['Label']
        features = self.features.copy()
        cat_feats = self.cat_feats.copy()
        for f in cat_feats:
            for df in [train, test]:
                df[f] = df[f].astype('category')

        oof = np.zeros(train.shape[0])
        predictions = np.zeros(test.shape[0])
        feature_importance_df = pd.DataFrame()
        folds = StratifiedKFold(n_splits=5, random_state=4584, shuffle=True)
        for fold, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
            print("Fold {}".format(fold))
            trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])
            val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])
            clf = lgb.train(
                params,
                trn_data,
                20000,
                valid_sets=[trn_data, val_data],
                verbose_eval=200,
                early_stopping_rounds=early_stop,
                categorical_feature=cat_feats,
            )
            oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = [self.map_columns[i] for i in features]
            fold_importance_df["importance"] = clf.feature_importance()
            fold_importance_df["fold"] = fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
        print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
        return oof, predictions, feature_importance_df

    def gen_submit(self, pred, name='submit'):
        """
        生成提交文件
        @param: pred:预测结果
        @param: name:提交文件名称
        """
        submit = self.submit.copy()
        tmp = self.test.copy()
        tmp['pred'] = pred
        del submit['Label']
        submit = pd.merge(submit, tmp[['ID', 'pred']], on='ID', how='left')
        submit.rename(columns={'pred': 'Label'}, inplace=True)
        submit.to_csv(name + '.csv', index=False)



model = BaseModel()

model.gen_dataset()

params = {
    'bagging_freq': 1,
    'bagging_fraction': 0.85,
    'bagging_seed': int(np.random.rand() * 100),
    'boost': 'gbdt',
    'feature_fraction': 0.85,     # 随机选择85%的特征
    'feature_fraction_seed': int(np.random.rand() * 100),
    'learning_rate': 0.01,
    'max_depth': 10,     # 貌似 8 改成10 提高了
    'metric': 'auc',
    'min_data_in_leaf': 15,    # 一个叶子上数据的最小数量. 可以用来处理过拟合. default 20
    'num_leaves': 28,   # 64 应该没有32好
    'num_threads': 4,
    'objective': 'binary',
    "lambda_l1": 0.5,
    'lambda_l2': 1.2,
    'verbosity': 1,
    'is_unbalance': True
}

oof, pred, feat_importance = model.model_train('LGB', params, seed=int(np.random.rand() * 100))

# 线下CV score:0.90713 线上0.923501

model.gen_submit(pred)

plt.figure(figsize=(16, 30))
sns.barplot(x="importance",
            y="Feature",
            data=(feat_importance.sort_values(by="importance", ascending=False)))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()