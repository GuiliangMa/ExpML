import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
# 读取数据集
train_df = pd.read_csv('../data/forAge/train.csv')
test_df = pd.read_csv('../data/forAge/test.csv')

# 保存测试集中乘客ID
test_passenger_ids = test_df['PassengerId']

# 为了特征工程，将训练集和测试集合并
combined_df = pd.concat([train_df, test_df], sort=False)

# 定义数值和分类特征
numerical_features = ['Pclass', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Sex', 'Cabin', 'Embarked', 'Title', 'TicketPrefix']

# 数值特征处理管道
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类特征处理管道
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 合并处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 应用预处理到合并后的数据集
combined_processed = preprocessor.fit_transform(combined_df.drop(columns=['PassengerId', 'Age', 'LastName']))

# 分离处理后的训练集和测试集
X_train_processed = combined_processed[:len(train_df)]
X_test_processed = combined_processed[len(train_df):]
y_train = train_df['Age']

print("Data is processed successfully\n")

# 定义模型
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Support Vector Regression': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': lgb.LGBMRegressor(
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        n_estimators=1000,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1
    ),
    'CatBoost': CatBoostRegressor(verbose=0),
    'Neural Network': MLPRegressor(max_iter=10000),
}

# 使用交叉验证评估每个模型
model_scores = {}
for name, model in models.items():
    print(f"{name} is working")
    scores = cross_val_score(model, X_train_processed, y_train, cv=5, scoring='neg_mean_squared_error')
    model_scores[name] = scores.mean()

print()
# 输出每个模型的得分情况
for name, score in model_scores.items():
    print(f"{name}: {score}")

# 选择负均方误差（负值）最大的模型，即误差最小的模型
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]
print(f"\n选择的最佳模型是: {best_model_name}，其负均方误差为: {model_scores[best_model_name]}")

# 在整个训练集上训练最佳模型
best_model.fit(X_train_processed, y_train)

# 预测测试集的年龄
y_test_pred = best_model.predict(X_test_processed)

# 将预测结果保留整数部分，小数部分统一填充为0.5
y_test_pred_int = y_test_pred.astype(int) + 0.5

# 准备结果
results = pd.DataFrame({'PassengerId': test_passenger_ids, 'Predicted_Age': y_test_pred_int})

# 保存结果到CSV文件
results.to_csv(r'../data/forAge/predicted.csv', index=False)
