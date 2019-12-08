
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder


import warnings
from jupyterthemes import jtplot


jtplot.style()
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


class BaseModel:
    def __init__(self):
        self.train = None
        self.test = None
        self.train_label = None
        self.features = None
        self.cat_feats = None

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
        print("train data shape:" + str(self.train.shape))
        print('Done!')

    def _missing_counter(self, df, cols=None):
        """
        统计数据缺失情况
        @param: df:数据集
        @param: cols:要统计缺失情况的特征列表
        """
        if cols is None:
            cols = df.columns
        counter = pd.DataFrame(columns=['Feature','Count','Percent'])
        counter['Feature'] = cols
        counter = counter.set_index('Feature')
        len = df.shape[0]
        for feat in cols:
            num = sum(df[feat].isna())
            percent = num / len
            counter['Count'].loc[feat] = num
            counter['Percent'].loc[feat] = percent
        return counter[counter['Count']!=0]

    def preprocess(self):
        """
        数据预处理（数据清洗等）
        """
        print('Preprocessing...', end='\t')
        # 处理缺失值

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
        for df in [train, test]:
            df['总税'] = df['印花税'] + df['增值税'] + df['企业所得税'] + df['城建税']
            df['注册资本税收比'] = df['注册资本'] / df['总税']
            # df['长期负债合计_差值'] = df['长期负债合计_年末数'] - df['长期负债合计_年末数']
            # df['长期借款_差值'] = df['长期借款_年末数'] - df['长期借款_年初数']
            # df['货币资金_差值'] = df['货币资金_年末数']- df['货币资金_年初数']
            # df['未分配利润_差值'] = df['未分配利润_年末数'] - df['未分配利润_年初数']
            # df['其他应收款_差值'] = df['其他应收款_年末数'] - df['其他应收款_年初数']
            # df['所有者权益合计_差值'] = df['所有者权益合计_年末数'] - df['所有者权益合计_年初数']
            # df['应收账款_差值'] = df['应收账款_年末数']- df['应收账款_年初数']

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

        print("Done!")

    def gen_dataset(self, path=''):
        """
        高级封装，数据读取+数据预处理+特征工程
        @param: path:数据存储的根目录
        """
        self.load_data(path)
        self.preprocess()
        self.feature_engineering()

    def model_train(self, model, params, seed, early_stop=200):
        """
        模型训练
        @param: model:模型类型
        @param: params:模型参数
        @param: seed:随机数种子
        @param: early_stop:模型训练时的早停参数
        """
        if model == 'XGB':
            oof, predictions, feature_importance_df = self._xgb_model(params, seed, early_stop)
        return oof, predictions, feature_importance_df

    def _xgb_model(self, params, seed=4545, num_rounds=None):
        """
        使用XGBOOST进行五折交叉训练
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
        folds = StratifiedKFold(n_splits=5, random_state=9816, shuffle=True)
        for fold, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
            print("Fold {}".format(fold))
            # model = XGBClassifier()
            plst = params.items()

            dtrain = xgb.DMatrix(train.iloc[trn_idx][features],target.iloc[trn_idx])

            if num_rounds is None:
                num_rounds = 100
            model = xgb.train(plst,dtrain,num_rounds)
            dtest = xgb.DMatrix(train.iloc[val_idx][features])
            predictions = model.predict(dtest)



            # model.fit(train.iloc[trn_idx][features],target.iloc[trn_idx])
            # trn_data = xgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])
            # val_data = xgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])
            # clf = xgb.train(
            #     params,
            #     trn_data,
            #     20000,
            #     valid_sets=[trn_data, val_data],
            #     verbose_eval=200,
            #     early_stopping_rounds=early_stop,
            #     categorical_feature=cat_feats,
            # )
            oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
            # fold_importance_df = pd.DataFrame()
            # fold_importance_df["Feature"] = [self.map_columns[i] for i in features]
            # fold_importance_df["importance"] = clf.feature_importance()
            # fold_importance_df["fold"] = fold + 1
            # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

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
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()

dtrain = xgb.DMatrix()
num_rounds = 100
clf = model.model_train('XGB',params,num_rounds)
pred = clf.predict()
# 线下CV score:0.90713 线上0.923501

model.gen_submit(pred)

plt.figure(figsize=(16, 30))
sns.barplot(x="importance",
            y="Feature",
            data=(feat_importance.sort_values(by="importance", ascending=False)))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()