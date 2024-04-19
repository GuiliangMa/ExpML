import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataProductor:

    @staticmethod
    def product(data_num, data_dim, data_path,flex=1):
        data = np.random.rand(data_num, data_dim)*flex
        columns = [f'x{i}' for i in range(data_dim)]
        df = pd.DataFrame(data,columns=columns)
        df.to_csv(data_path, index=False)