import pandas as pd

std = pd.read_csv('../data/answer/titanic/ground_truth.csv')
submit = pd.read_csv('../data/submit/pca_bp_t5.csv')

total = 419
diff_count = (std['Survived'] != submit['Survived']).sum()

print(f"Acc = {1-diff_count/total}")
