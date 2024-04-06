import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DataProcessor:
    def __init__(self, data_frame):
        self.data = data_frame

    def __fillNan(self, column, typ=0, default_value=0):
        if typ not in [0, 1, 2, 3]:
            raise ValueError("type must be 0, 1, 2, or 3.")
        if typ == 0:
            column.fillna(column.mode()[0], inplace=True)
        if typ == 1:
            column.fillna(column.median(), inplace=True)
        if typ == 2:
            column.fillna(column.mean(), inplace=True)
        if typ == 3:
            column.fillna(default_value, inplace=True)
        return column

    class_Mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    employer_type_Mapping = {'幼教与中小学校': 0, '政府机构': 1, '上市企业': 2, '普通企业': 3, '世界五百强': 4,
                             '高等教育机构': 5}
    industry_Mapping = {'金融业': 0, '国际组织': 1, '文化和体育业': 2, '建筑业': 3, '电力、热力生产供应业': 4,
                        '批发和零售业': 5, '采矿业': 6,
                        '房地产业': 7, '交通运输、仓储和邮政业': 8, '公共服务、社会组织': 9, '住宿和餐饮业': 10,
                        '信息传输、软件和信息技术服务业': 11, '农、林、牧、渔业': 12, '制造业': 13}
    work_year_Mapping = {'8 years': 8, '10+ years': 10, '3 years': 3, '7 years': 7, '4 years': 4, '1 year': 1,
                         '< 1 year': 0, '6 years': 6, '2 years': 2, '9 years': 9, '5 years': 5}

    def __dealString(self, column, mapping):
        values = column.values.tolist()
        for value in values:
            if value not in mapping:
                raise ValueError(f"Value '{value}' is not in {list(mapping.keys())}")
        column = column.map(mapping)
        return column

    def __calcMon2Target(self, date, target_year, target_month):
        target_date = datetime(target_year, target_month, 1).date()
        diff = relativedelta(date, target_date)
        total_months = total_months = diff.years * 12 + diff.months
        return total_months

    def __dealDate2Month(self, column):
        column = pd.to_datetime(column, format='%Y/%m/%d')
        column = column.apply(lambda x: self.__calcMon2Target(x, 1970, 1))
        return column

    def matchMonth(self, text):
        pattern = re.compile(r'[A-Za-z]+')
        matches = pattern.findall(text)
        try:
            month = datetime.strptime(matches[0], "%b").month
        except ValueError:
            raise ValueError(f"Value '{matches[0]}' is a valid month")
        return month

    def __dealString2Month(self, column):
        column = column.apply(lambda x: self.matchMonth(x))
        return column

    def Process(self):
        # 删除若干列,从训练集和测试集可以初步看出
        # loan_id和user_id是基本不重复的，因此可以删去。当前数据下policy_code下均为1，也可删去。
        self.data = self.data.drop(['loan_id'], axis=1)
        self.data = self.data.drop(['user_id'], axis=1)
        self.data = self.data.drop(['policy_code'], axis=1)

        # 默认均采用众数来填充所有缺失值
        for columnName in self.data.columns:
            self.data[columnName] = self.__fillNan(self.data[columnName])

        # 先处理字符串变为数值型
        # 第一步处理 class 属性
        self.data['class'] = self.__dealString(self.data['class'], self.class_Mapping)
        # 第二步处理 employer_type 属性
        self.data['employer_type'] = self.__dealString(self.data['employer_type'], self.employer_type_Mapping)
        # 第三步处理 industry 属性
        self.data['industry'] = self.__dealString(self.data['industry'], self.industry_Mapping)
        # 第四步处理 work_year 属性
        self.data['work_year'] = self.__dealString(self.data['work_year'], self.work_year_Mapping)
        # 第五步处理 issue_date 属性，该属性的含义 贷款发放的月份,因此只用截至到年份
        self.data['issue_date'] = self.__dealDate2Month(self.data['issue_date'])
        # 第六步处理 earlies_credit_mon 属性，该属性约束到月份
        self.data['earlies_credit_mon'] = self.__dealString2Month(self.data['earlies_credit_mon'])

        y_process = self.data['isDefault']
        x_process = self.data.drop(['isDefault'], axis=1)
        return x_process, y_process


if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    dp = DataProcessor(df)
    X, y = dp.Process()
    X.to_csv('../process/Processed_TrainData.csv', index=False)
