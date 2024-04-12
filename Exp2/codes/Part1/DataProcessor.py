import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def PrepareForTraining(df):
    df.insert(0,'',1)
    return df

def LinearBoundary(weights,x):
    y = (-weights[1] / weights[2]) * x - weights[0] / weights[2]
    return y

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))
