import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def findSegment(new_value, bin_edges):
    # 确保新值不超出bin_edges的范围
    if new_value < bin_edges[0]:
        return 0
    elif new_value >= bin_edges[-1]:  # 简化为bin_edges[-1]，表示最后一个元素
        return len(bin_edges)-2
    # 找到新值对应的段索引
    bin_index = np.digitize([new_value], bin_edges, right=False)[0] - 1
    return bin_index

def DataSegment(column,typ='Rice',bin_edges = []):
    df[column_name] = pd.cut(df[column_name], bins=bins, labels=range(k_rice), include_lowest=True)


df = pd.read_csv('../data/train.csv')
n = len(df['total_loan'])
k_rice = int(2*(n**(1/3)))
column_name = 'total_loan'

bins = np.linspace(df[column_name].min(), df[column_name].max(), k_rice + 1)
df[column_name] = pd.cut(df[column_name], bins=bins, labels=range(k_rice), include_lowest=True)

print(bins)

x = findSegment(47272.72727,bins)
print(x)

df.to_csv('../process/test.csv', index=False)


