# 基于sklearn 的 PCA 包进行白化的操作
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train_df = pd.read_csv('../data/preTrain/train.csv')
test_df = pd.read_csv('../data/preTrain/test.csv')
print(train_df.shape)
print(test_df.shape)

train_label = train_df['Survived'].copy()
train_df.drop('Survived', axis=1, inplace=True)
train_lastname = train_df['LastName'].copy()
train_df.drop('LastName', axis=1, inplace=True)
train_pid = train_df['PassengerId'].copy()
train_df.drop('PassengerId', axis=1, inplace=True)

test_lastname = test_df['LastName'].copy()
test_df.drop('LastName', axis=1, inplace=True)
test_pid = test_df['PassengerId'].copy()
test_df.drop('PassengerId', axis=1, inplace=True)

train_df['is_train'] = 1
test_df['is_train'] = 0

combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_data = combined_df.drop(columns=['is_train'])

scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined_data)
pca = PCA(whiten=True, n_components=0.95)
combined_pca = pca.fit_transform(combined_scaled)

combined_pca_df = pd.DataFrame(combined_pca, columns=[f'PC{i + 1}' for i in range(combined_pca.shape[1])])

# 重新加上标识符列
combined_pca_df['is_train'] = combined_df['is_train'].values

# 分离数据
train_pca_df = combined_pca_df[combined_pca_df['is_train'] == 1].drop(columns=['is_train'])
train_pca_df['Survived'] = train_label
train_pca_df['PassengerId'] = train_pid
train_pca_df['LastName'] = train_lastname

test_pca_df = combined_pca_df[combined_pca_df['is_train'] == 0].drop(columns=['is_train'])
test_pca_df['PassengerId'] = test_pid
test_pca_df['LastName'] = test_lastname

print("--------------------")
print(train_pca_df.shape)
print(test_pca_df.shape)

train_pca_df.to_csv('../data/preTrain/train_pca95.csv', index=False)
test_pca_df.to_csv('../data/preTrain/test_pca95.csv', index=False)