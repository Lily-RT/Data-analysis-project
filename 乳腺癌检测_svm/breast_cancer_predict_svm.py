import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics

# 加载数据
data = pd.read_csv('C:/Users/O-O RT/Desktop/数据分析相关书籍/项目/乳腺癌检测_svm/data.csv')
# 数据探索
pd.set_option('display.max_columns', None)
print(data.columns)
print(data.head())
print(data.describe())
# 将特征字段分成3组
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])
# 数据清洗
# ID列没用，删除该列
data.drop('id', axis=1, inplace=True)
# 将diagnosis列中的B良性用0表示，M恶性用1表示
data['diagnosis'] = data['diagnosis'].map({'B':0, 'M':1})
# 将肿瘤诊断结果可视化
sns.countplot(data['diagnosis'], label='Count')
# 用热力图显示——mean（均值）
corr1 = data[features_mean].corr()
plt.figure(figsize=(10, 8))
# annot=True显示每个方格的数据
sns.heatmap(corr1, annot=True)
plt.show()
# 热力图显示——se（标准差）
corr2 = data[features_se].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr2, annot=True)
plt.show()
# 热力图显示——worst（最大值）
corr3 = data[features_worst].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr3, annot=True)
plt.show()
# 特征集选择
features_remain = ['radius_mean', 'texture_mean', 'compactness_mean', 'smoothness_mean', 'symmetry_mean',
                   'fractal_dimension_mean', 'radius_se', 'texture_se', 'compactness_se', 'smoothness_se',
                   'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'compactness_worst',
                   'smoothness_worst', 'symmetry_worst', 'fractal_dimension_worst']
# 抽取33% 的数据作为测试集，其余为训练集
train, test = train_test_split(data, test_size=0.33, random_state=0)
# 抽取特征选择的数值作为训练和测试数据
train_x = train[features_remain]
train_y = train['diagnosis']
test_x = test[features_remain]
test_y = test['diagnosis']
# 采用Z-score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)
# 创建SVM分类器
model = svm.SVC()
# 用训练集做训练
model.fit(train_x, train_y)
# 用测试集做预测
prediction = model.predict(test_x)
print('svm准确率：', metrics.accuracy_score(prediction, test_y))