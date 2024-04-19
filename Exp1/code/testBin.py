from DataProcess import DataProcessor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def equal_width_binning(data, num_bins):
    min_val = np.min(data)
    max_val = np.max(data)
    width = (max_val - min_val) / num_bins
    bins = np.linspace(min_val, max_val, num_bins + 1)  # 使用 linspace 保证最大值包括在内
    data_binned = np.digitize(data, bins, right=True)  # right=True 以包括右边界
    return data_binned, bins


def model_fit_and_log_likelihood(data, data_binned, num_bins):
    means = np.array([
        data[data_binned == i].astype(float).mean() if len(data[data_binned == i]) > 0 else np.mean(data.astype(float))
        for i in range(1, num_bins + 1)
    ])
    likelihoods = np.array([
        -0.5 * np.sum((data[data_binned == i].astype(float) - means[i - 1]) ** 2)
        if len(data[data_binned == i]) > 0 else 0
        for i in range(1, num_bins + 1)
    ])
    log_likelihood = np.sum(likelihoods)
    return log_likelihood


# AIC 和 BIC 计算
def calculate_aic_bic(data, num_bins):
    data_binned, bins = equal_width_binning(data, num_bins)
    log_likelihood = model_fit_and_log_likelihood(data, data_binned, num_bins)
    k = num_bins
    n = len(data)
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood
    return aic, bic


def count(data):
    num_bins_range = range(1, 21)
    results = [calculate_aic_bic(data, num_bins) for num_bins in num_bins_range]

    # 选择最佳箱数
    best_aic, best_bic = np.min([res[0] for res in results]), np.min([res[1] for res in results])
    best_num_bins_aic = num_bins_range[np.argmin([res[0] for res in results])]
    best_num_bins_bic = num_bins_range[np.argmin([res[1] for res in results])]

    print("Best number of bins by AIC:", best_num_bins_aic, "with AIC:", best_aic)
    print("Best number of bins by BIC:", best_num_bins_bic, "with BIC:", best_bic)
    return best_num_bins_aic, best_num_bins_bic


dataframe = pd.read_csv('../data/train.csv')
print(list(dataframe.columns))
for col in list(dataframe.columns):
    print("'"+col+"'"+":13,")
exit(0)
dp = DataProcessor(dataframe)
X, y = dp.Process()
X_old = dataframe.drop(['isDefault'], axis=1)
segmentMap = {
    'total_loan': {'num': 15, 'type': 'w'},
    'monthly_payment': {'num': 10, 'type': 'w'},
    'issue_date': {'num': 14, 'type': 'w'},
    'debt_loan_ratio': {'num': 4, 'type': 'f'},
    'scoring_low': {'num': 20, 'type': 'w'},
    'scoring_high': {'num': 14, 'type': 'w'},
    'recircle_b': {'num': 10, type: 'f'},
    'recircle_u': {'num': 10, type: 'w'},
    'f0': {'num': 10, type: 'f'},
    'f2': {'num': 5, type: 'f'},
    'f3': {'num': 5, type: 'f'},
    'f4': {'num': 5, type: 'f'},
    'early_return_amount': {'num': 4, type: 'f'},
    'early_return_amount_3mon': {'num': 3, type: 'f'},
}
# 先基于等宽分箱考虑


# 'interest':分7段后与class基本一致，并且class和interest是强相关,class表示贷款级别而interest是贷款利率，理论上可以二选一，否则采取7-10应该均可
# 'post_code' 考虑到该属性为邮政编码，这样看来不应进行分段可能更好，可能8,9,10不失为一个好的选择。与y的相关性很低，为0.005489565192954676
# 'region',该属性的代表值为地区编码，不清除其相邻地区编号是否有内部含义。相关性0.011347982059768036，可以尝试删去
# 'title',不明晰该类别的有用性，极度偏态，可以考虑该列与y的相关性，如果很低可以忽略，相关性很低-0.003888648046220826
# 'known_outstanding_loan',为一个取47的疑似连续值，取10或者其他？(先不添加—，后续增加）

# 'total_loan':15 or 16 or 17
# 'monthly_payment',在进行原图的柱状图绘制时，其自动绘制出接近10个点的分布，然后根据对比可知，9，10，11看上去都还可以，50组时出现部分数据采样为0
# 'issue_date',采用14时整体分布相似，其余情况下12-15也都可以
# 'debt_loan_ratio',该属性极度偏斜，采用等宽分布应该不合适,先采用等频分4段尝试
# 'scoring_low',采取20可能合理，30疑似也可以
# 'scoring_high',采用14或者更低，其分布为单峰，可以再调节

# 'recircle_b',该类偏态也极其严重，采用等频或等深
# 'recircle_u', 采用10为单峰分布

# 'f0',略有偏态，考虑是否可以不分段
# 'f2',偏态十分严重，考虑是否要等频分类
# 'f3',偏态十分严重，考虑要等频分类
# 'f4',偏态十分严重，考虑等频分类
# 'early_return_amount',偏态严重，也需要考虑等频分箱，或者考虑分成三段
# 'early_return_amount_3mon',感官可以采用等频或者二分类

# theta = dataframe['title'].corr(dataframe['isDefault'], method='pearson')
# print(theta)
# exit(0)

columnName = 'total_loan'
preX = X[columnName]

column = X[columnName]
# print(column.value_counts().sort_index())
print(len(column.value_counts()))
num, _ = count(column)
print(num)
bins = np.linspace(column.min(), column.max(), num + 1)

# cuts = pd.qcut(column, q=num, duplicates='drop')
# bins = [interval.left for interval in cuts.cat.categories] + [cuts.cat.categories[-1].right]
# num = len(bins) - 1
# print(num)
# print(bins)

postX = pd.cut(column, bins=bins, labels=range(num), include_lowest=True)
# print(postX.value_counts())

plt.subplot(1, 2, 1)
value_counts = preX.value_counts().sort_index()
value_counts.plot(kind='bar')
plt.xlabel('values')
plt.xticks([])
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
value_counts = postX.value_counts().sort_index()
value_counts.plot(kind='bar')
plt.xlabel('values')
plt.xticks([])
plt.ylabel('Frequency')

plt.show()
