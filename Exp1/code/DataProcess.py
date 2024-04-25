import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DataProcessor:
    def __init__(self, dataFrame,segmentMap, fillNanRules={}):
        '''
        :param segmentMap: 分段分箱列表
        :param fillNanRules: 填充规则 name:[method,defaultValue]这样的字典
        '''
        self.data = dataFrame
        self.fillNanRules = fillNanRules
        self.isProcessed = False
        self.segmentMap = segmentMap
        self.Segment = {}

    def __fillNan(self, column, typ=0, default_value=0):
        '''
        :param typ: 0 为众数、1为中位数、2为平均数、3为填充为default_value值
        '''
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

    @staticmethod
    def dropColumn(data):
        data = data.drop(['isDefault'], axis=1)

        # 删除若干列,从训练集和测试集可以初步看出
        # loan_id和user_id是基本不重复的，因此可以删去。当前数据下policy_code下均为1，也可删去。
        data = data.drop(['loan_id'], axis=1)
        data = data.drop(['user_id'], axis=1)
        data = data.drop(['policy_code'], axis=1)

        # interest 与 class的强相关
        # ('interest', 'class'): 0.9254597054479434
        data = data.drop(['interest'], axis=1)

        # ('f3', 'f4'): 0.8438089877232243
        data = data.drop(['f4'], axis=1)

        # ('early_return_amount', 'early_return_amount_3mon'): 0.7530913899047247
        data = data.drop(['early_return_amount'], axis=1)

        # ('scoring_low', 'scoring_high'): 0.8890661841570701
        data = data.drop(['scoring_low'], axis=1)

        # ('total_loan', 'monthly_payment'): 0.9256103360334527
        data = data.drop(['total_loan'], axis=1)
        return data

    def fillNanMethod(self, data):
        # 默认均采用众数来填充所有缺失值
        for columnName in data.columns:
            if columnName in self.fillNanRules:
                if self.fillNanRules[columnName][0] == 3:
                    data[columnName] = self.__fillNan(data[columnName], self.fillNanRules[columnName][0],
                                                      self.fillNanRules[columnName][1])
                else:
                    data[columnName] = self.__fillNan(data[columnName], self.fillNanRules[columnName][0])
            else:
                data[columnName] = self.__fillNan(data[columnName])
        return data

    Mapping = {'class': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
               'employer_type': {'幼教与中小学校': 0, '政府机构': 1, '上市企业': 2, '普通企业': 3, '世界五百强': 4,
                                 '高等教育机构': 5},
               'industry': {'金融业': 0, '国际组织': 1, '文化和体育业': 2, '建筑业': 3, '电力、热力生产供应业': 4,
                            '批发和零售业': 5, '采矿业': 6,
                            '房地产业': 7, '交通运输、仓储和邮政业': 8, '公共服务、社会组织': 9, '住宿和餐饮业': 10,
                            '信息传输、软件和信息技术服务业': 11, '农、林、牧、渔业': 12, '制造业': 13},
               'work_year': {'8 years': 8, '10+ years': 10, '3 years': 3, '7 years': 7, '4 years': 4, '1 year': 1,
                             '< 1 year': 0, '6 years': 6, '2 years': 2, '9 years': 9, '5 years': 5}}

    @staticmethod
    def __dealString(column, mapping):
        values = column.values.tolist()
        for value in values:
            if value not in mapping:
                raise ValueError(f"Value '{value}' is not in {list(mapping.keys())}")
        column = column.map(mapping)
        return column

    @staticmethod
    def __calcMon2Target(date, target_year, target_month):
        target_date = datetime(target_year, target_month, 1).date()
        diff = relativedelta(date, target_date)
        total_months = total_months = diff.years * 12 + diff.months
        return total_months

    def __dealDate2Month(self, column):
        column = pd.to_datetime(column, format='%Y/%m/%d')
        column = column.apply(lambda x: self.__calcMon2Target(x, 1970, 1))
        return column

    def __dealString2Month(self, column):
        column = column.apply(lambda x: self.matchMonth(x))
        return column

    @staticmethod
    def matchMonth(text):
        pattern = re.compile(r'[A-Za-z]+')
        matches = pattern.findall(text)
        try:
            month = datetime.strptime(matches[0], "%b").month
        except ValueError:
            raise ValueError(f"Value '{matches[0]}' is a valid month")
        return month

    def dealStringMethod(self, data):
        data['class'] = self.__dealString(data['class'], self.Mapping['class'])
        # 第二步处理 employer_type 属性
        data['employer_type'] = self.__dealString(data['employer_type'], self.Mapping['employer_type'])
        # 第三步处理 industry 属性
        data['industry'] = self.__dealString(data['industry'], self.Mapping['industry'])
        # 第四步处理 work_year 属性
        data['work_year'] = self.__dealString(data['work_year'], self.Mapping['work_year'])
        # 第五步处理 issue_date 属性，该属性的含义 贷款发放的月份,因此只用截至到年份
        data['issue_date'] = self.__dealDate2Month(data['issue_date'])
        # 第六步处理 earlies_credit_mon 属性，该属性约束到月份
        data['earlies_credit_mon'] = self.__dealString2Month(data['earlies_credit_mon'])
        return data

    @staticmethod
    def calculate_within_bin_variance(data, bins):
        hist, bin_edges = np.histogram(data, bins=bins)
        within_variances = []
        for i in range(len(bin_edges) - 1):
            bin_data = data[(data >= bin_edges[i]) & (data < bin_edges[i + 1])]
            if len(bin_data) > 1:
                within_variances.append(np.var(bin_data))
            else:
                within_variances.append(0)
        return np.mean(within_variances) if within_variances else float('inf')

    @staticmethod
    def calculate_between_bin_variance(data, bins):
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_means = []
        for i in range(len(bin_edges) - 1):
            bin_data = data[(data >= bin_edges[i]) & (data < bin_edges[i + 1])]
            if len(bin_data) > 0:
                bin_means.append(np.mean(bin_data))
        overall_mean = np.mean(data)
        between_variance = sum(
            [(mean - overall_mean) ** 2 * len(data[(data >= bin_edges[i]) & (data < bin_edges[i + 1])]) for i, mean in
             enumerate(bin_means)])
        return between_variance / len(data)

    def find_optimal_bins(self, data, max_bins=90):
        best_bins = 2
        best_score = float('-inf')
        for bins in range(2, max_bins + 1):
            within_variance = self.calculate_within_bin_variance(data, bins)
            between_variance = self.calculate_between_bin_variance(data, bins)
            score = between_variance - within_variance  # Aim for low within and high between
            if score > best_score:
                best_score = score
                best_bins = bins
            # print(f"Bins: {bins}, Score: {score}, Within Variance: {within_variance}, Between Variance: {between_variance}")
        return best_bins

    def __dealDataSegment(self, column, column_name, typ='train'):
        if column_name in self.Segment:
            column = pd.cut(column, bins=self.Segment[column_name], labels=range(len(self.Segment[column_name]) - 1),
                            include_lowest=True)
            return column
        if typ == 'test':
            return column
        if column_name not in self.segmentMap:
            return column

        num = self.segmentMap[column_name]
        bins = np.linspace(column.min(), column.max(), num + 1)
        column = pd.cut(column, bins=bins, labels=range(num),
                        include_lowest=True)
        self.Segment[column_name] = bins
        return column

        # length = len(column.value_counts().index)
        # if 50 <= length <= 2500:
        #     num = 1 + int(np.log2(length))
        #     bins = np.linspace(column.min(), column.max(), num + 1)
        #     column = pd.cut(column, bins=bins, labels=range(num),
        #                     include_lowest=True)
        #     self.Segment[column_name] = bins
        #     return column
        # elif length > 2500:
        #     # num = int(length / 100)
        #     num = 2 * int(length ** (1 / 3))
        #     bins = np.linspace(column.min(), column.max(), num + 1)
        #     column = pd.cut(column, bins=bins, labels=range(num),
        #                     include_lowest=True)
        #     self.Segment[column_name] = bins
        #     return column
        # return column

    def dealSegment(self, data, typ='train'):
        for columnName in data.columns:
            data[columnName] = self.__dealDataSegment(data[columnName], columnName, typ=typ)
        # print(self.Segment)
        return data

    def Process(self):
        self.isProcessed = True
        y_process = self.data['isDefault']
        self.data = self.dropColumn(self.data)
        self.data = self.fillNanMethod(self.data)
        self.data = self.dealStringMethod(self.data)
        self.data = self.dealSegment(self.data)
        X_process = self.data
        print("Process")
        return X_process, y_process

    def Deal(self, df):
        print("Deal For Test")
        if self.isProcessed is False:
            raise ValueError(f'This DataProcessor is not processed or trained')
        y_process = df['isDefault']
        # print("Test Drop")
        df = self.dropColumn(df)
        # print("Test Fill")
        df = self.fillNanMethod(df)
        # print("Test Trans")
        df = self.dealStringMethod(df)
        # print("Test Segment")
        df = self.dealSegment(df, 'test')
        X_process = df
        # print("Test Data Finished")
        return X_process, y_process


segmentMap = {
        'post_code': 13,
        'title': 13,
        'known_outstanding_loan': 13,
        'monthly_payment': 13,
        'issue_date': 13,
        'debt_loan_ratio': 13,
        'scoring_high': 13,
        'recircle_b': 13,
        'recircle_u': 13,
        'f0': 13,
        'f2': 13,
        'f3': 13,
        'early_return_amount': 13,
    }

if __name__ == '__main__':
    dataframe = pd.read_csv('../data/train.csv')
    dp = DataProcessor(dataframe,segmentMap)
    X, y = dp.Process()
    X.to_csv('../process/Processed_Train_Full_Data.csv', index=False)
