import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DataProcessor:
    def __init__(self, dataFrame, fillNanRules={}):
        self.data = dataFrame
        self.fillNanRules = fillNanRules

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

    Mapping = {'class': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
               'employer_type': {'幼教与中小学校': 0, '政府机构': 1, '上市企业': 2, '普通企业': 3, '世界五百强': 4,
                                 '高等教育机构': 5},
               'industry': {'金融业': 0, '国际组织': 1, '文化和体育业': 2, '建筑业': 3, '电力、热力生产供应业': 4,
                            '批发和零售业': 5, '采矿业': 6,
                            '房地产业': 7, '交通运输、仓储和邮政业': 8, '公共服务、社会组织': 9, '住宿和餐饮业': 10,
                            '信息传输、软件和信息技术服务业': 11, '农、林、牧、渔业': 12, '制造业': 13},
               'work_year': {'8 years': 8, '10+ years': 10, '3 years': 3, '7 years': 7, '4 years': 4, '1 year': 1,
                             '< 1 year': 0, '6 years': 6, '2 years': 2, '9 years': 9, '5 years': 5}}

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

    Segment = {}

    def __dealDataSegment(self, column_name):
        # print(column_name)
        column = self.data[column_name]
        if column.dtype == 'region':
            return column
        if column_name in self.Segment:
            column = pd.cut(column, bins=self.Segment[column_name], labels=range(len(self.Segment[column_name]) - 1),
                            include_lowest=True)
            return column
        length = len(column.value_counts().index)
        if 50 <= length <= 2500:
            num = 1 + int(np.log2(length))
            bins = np.linspace(column.min(), column.max(), num + 1)
            column = pd.cut(column, bins=bins, labels=range(num),
                            include_lowest=True)
            self.Segment[column_name] = bins
            return column
        elif 2500 < length <= len(column):
            num = 2*int(length ** (1 / 3))
            # print(num)
            bins = np.linspace(column.min(), column.max(), num + 1)
            column = pd.cut(column, bins=bins, labels=range(num),
                            include_lowest=True)
            self.Segment[column_name] = bins
            return column
        return column

    def __dealTestDataSegment(self, df, column_name):
        column = df[column_name]
        if column_name in self.Segment:
            column = pd.cut(column, bins=self.Segment[column_name], labels=range(len(self.Segment[column_name]) - 1),
                            include_lowest=True)
            return column
        return column

    def __dealString2Month(self, column):
        column = column.apply(lambda x: self.matchMonth(x))
        return column

    def Process(self):
        y_process = self.data['isDefault']
        self.data = self.data.drop(['isDefault'], axis=1)

        # 删除若干列,从训练集和测试集可以初步看出
        # loan_id和user_id是基本不重复的，因此可以删去。当前数据下policy_code下均为1，也可删去。
        self.data = self.data.drop(['loan_id'], axis=1)
        self.data = self.data.drop(['user_id'], axis=1)
        self.data = self.data.drop(['policy_code'], axis=1)
        # interest 与 class的强相关
        self.data = self.data.drop(['interest'], axis=1)
        self.data = self.data.drop(['f4'], axis=1)
        self.data = self.data.drop(['early_return_amount'],axis=1)

        # 默认均采用众数来填充所有缺失值
        for columnName in self.data.columns:
            if columnName in self.fillNanRules:
                if self.fillNanRules[columnName][0] == 3:
                    self.data[columnName] = self.__fillNan(self.data[columnName], self.fillNanRules[columnName][0],
                                                           self.fillNanRules[columnName][1])
                else:
                    self.data[columnName] = self.__fillNan(self.data[columnName], self.fillNanRules[columnName][0])
            else:
                self.data[columnName] = self.__fillNan(self.data[columnName])

        # 先处理字符串变为数值型
        # 第一步处理 class 属性
        self.data['class'] = self.__dealString(self.data['class'], self.Mapping['class'])
        # 第二步处理 employer_type 属性
        self.data['employer_type'] = self.__dealString(self.data['employer_type'], self.Mapping['employer_type'])
        # 第三步处理 industry 属性
        self.data['industry'] = self.__dealString(self.data['industry'], self.Mapping['industry'])
        # 第四步处理 work_year 属性
        self.data['work_year'] = self.__dealString(self.data['work_year'], self.Mapping['work_year'])
        # 第五步处理 issue_date 属性，该属性的含义 贷款发放的月份,因此只用截至到年份
        self.data['issue_date'] = self.__dealDate2Month(self.data['issue_date'])
        # 第六步处理 earlies_credit_mon 属性，该属性约束到月份
        self.data['earlies_credit_mon'] = self.__dealString2Month(self.data['earlies_credit_mon'])

        # 再处理分箱（分段）
        # 基于数据统计，大致按照如下分箱方式进行分箱，且分箱后其对应的元素的代表值为箱号
        # 针对数值100-1000种的列采用 Sturges，1000-4000的列采用Rice，4000-9000采用Freedman-Diaconis
        # 并且进行灵活处理，可以提前设置分段来传入DataProcessor
        for columnName in self.data.columns:
            self.data[columnName] = self.__dealDataSegment(columnName)

        x_process = self.data
        return x_process, y_process

    def Deal(self, df):
        y_process = df['isDefault']
        df = df.drop(['isDefault'], axis=1)

        # 删除若干列,从训练集和测试集可以初步看出
        # loan_id和user_id是基本不重复的，因此可以删去。当前数据下policy_code下均为1，也可删去。
        df = df.drop(['loan_id'], axis=1)
        df = df.drop(['user_id'], axis=1)
        df = df.drop(['policy_code'], axis=1)
        df = df.drop(['interest'], axis=1)
        df = df.drop(['f4'], axis=1)
        df = df.drop(['early_return_amount'], axis=1)

        # 默认均采用众数来填充所有缺失值
        for columnName in df.columns:
            if columnName in self.fillNanRules:
                if self.fillNanRules[columnName][0] == 3:
                    df[columnName] = self.__fillNan(df[columnName], self.fillNanRules[columnName][0],
                                                    self.fillNanRules[columnName][1])
                else:
                    df[columnName] = self.__fillNan(df[columnName], self.fillNanRules[columnName][0])
            else:
                df[columnName] = self.__fillNan(df[columnName])

        # 先处理字符串变为数值型
        # 第一步处理 class 属性
        df['class'] = self.__dealString(df['class'], self.Mapping['class'])
        # 第二步处理 employer_type 属性
        df['employer_type'] = self.__dealString(df['employer_type'], self.Mapping['employer_type'])
        # 第三步处理 industry 属性
        df['industry'] = self.__dealString(df['industry'], self.Mapping['industry'])
        # 第四步处理 work_year 属性
        df['work_year'] = self.__dealString(df['work_year'], self.Mapping['work_year'])
        # 第五步处理 issue_date 属性，该属性的含义 贷款发放的月份,因此只用截至到年份
        df['issue_date'] = self.__dealDate2Month(df['issue_date'])
        # 第六步处理 earlies_credit_mon 属性，该属性约束到月份
        df['earlies_credit_mon'] = self.__dealString2Month(df['earlies_credit_mon'])
        x_process = df

        # 再处理分箱（分段）
        # 基于数据统计，大致按照如下分箱方式进行分箱，且分箱后其对应的元素的代表值为箱号
        # 针对数值100-1000种的列采用 Sturges，1000-4000的列采用Rice
        # 并且进行灵活处理，可以提前设置分段来传入DataProcessor

        for columnName in df.columns:
            df[columnName] = self.__dealTestDataSegment(df, columnName)
        return x_process, y_process



if __name__ == '__main__':
    dataframe = pd.read_csv('../data/train.csv')
    dp = DataProcessor(dataframe)
    X, y = dp.Process()
    X.to_csv('../process/Processed_TrainData.csv', index=False)
