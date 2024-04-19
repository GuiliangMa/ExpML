import time
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from dataProductor import DataProductor


def TSP_emulate(data):
    start_time = time.time()

    num_rows = data.shape[0]
    index_permutations = list(permutations(range(num_rows)))
    max_perm = []
    min_dis = np.inf
    for perm in index_permutations:
        index = list(perm)
        index2 = index[1:]
        index2.append(index[0])
        distances = np.sqrt(np.sum((data[index, :] - data[index2, :]) ** 2, axis=1))  # 计算每行的欧氏距离
        total_distance = np.sum(distances)
        if total_distance < min_dis:
            min_dis = total_distance
            max_perm = perm

    end_time = time.time()
    elapsed_time = (end_time - start_time)
    print('Emulate Elapsed Time: ', elapsed_time)
    return data[list(max_perm), :], min_dis


def initialize_nodes(num_nodes, flex=1):
    theta = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    return np.array([np.cos(theta) + 0.5 * flex, np.sin(theta) + 0.5 * flex]).T


def initialize_nodes_out(num_nodes, cities, flex=1):
    center = np.array([0.5*flex,0.5*flex])
    dis = np.linalg.norm(cities-center, axis=1)
    index = np.argmax(dis)
    max_dis = dis[index]
    theta = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    return np.array([(np.cos(theta) + 0.5 * flex)*(int(max_dis+1)),
                     (np.sin(theta) + 0.5 * flex)*(int(max_dis+1))]).T




def find_winner(city, nodes):
    dis = np.linalg.norm(nodes - city, axis=1)
    return np.argmin(dis)


def update_nodes(nodes, winner_index, city, alpha, neighborhood_size):
    num_nodes = len(nodes)
    for index in range(num_nodes):
        ring_dis = min(np.abs(index - winner_index), num_nodes - np.abs(index - winner_index))
        if ring_dis <= neighborhood_size:
            influence = np.exp(-ring_dis ** 2 / (2 * (neighborhood_size ** 2)))
            nodes[index] += alpha * influence * (city - nodes[index])


def SOM_TSP(cities, num_nodes, epochs, initial_alpha, initial_neighborhood_size, flex=1):
    start_time = time.time()
    # nodes = initialize_nodes(num_nodes, flex)
    nodes = initialize_nodes_out(num_nodes, cities, flex)
    for epoch in range(epochs):
        np.random.shuffle(cities)
        alpha = initial_alpha * (1 - epoch / epochs)
        neighborhood_size = initial_neighborhood_size * (1 - epoch / epochs)
        for city in cities:
            winner_index = find_winner(city, nodes)
            update_nodes(nodes, winner_index, city, alpha, neighborhood_size)
    end_time = time.time()
    elapsed_time = (end_time - start_time)
    print('SOM Elapsed Time: ', elapsed_time)
    nodes2 = np.append(nodes[1:], nodes[0]).reshape([len(nodes), 2])
    distances = np.linalg.norm(nodes - nodes2, axis=1)
    total_distance = np.sum(distances)
    return nodes, total_distance


def plotTSP(pic_data, dis, num_cities, type, k=0, save_path_name=None):
    num = pic_data.shape[0]
    pic_data = np.append(pic_data, pic_data[0]).reshape([num + 1, 2])
    plt.plot(pic_data[:, 0], pic_data[:, 1], 'go')  # 'ro' 是红色的圆点

    # 添加箭头
    for i in range(len(pic_data) - 1):
        arrow = FancyArrowPatch((pic_data[i, 0], pic_data[i, 1]),
                                (pic_data[i + 1, 0], pic_data[i + 1, 1]),
                                arrowstyle='-', mutation_scale=20,
                                color='b', linewidth=2)
        plt.gca().add_patch(arrow)
    if type == 'Emulate':
        plt.title(f'type={type}\ndis={dis}\nnum_cities={num_cities}')
    if type == "SOM":
        plt.title(f'type={type}\ndis={dis}\nk={k},num_cities={num_cities}')
    if save_path_name is not None:
        plt.savefig(save_path_name)
    plt.show()


if __name__ == '__main__':
    flex_size = 10
    # DataProductor.product(data_num=10, data_dim=2, data_path='../data/data.csv', flex=flex_size)
    data = pd.read_csv('../data/data.csv')
    num = data.shape[0]

    data = np.array(data)

    # pic_data, tsp_dis = TSP_emulate(data)
    # plt.plot(data[:, 0], data[:, 1], 'r.', markersize=20)
    # plotTSP(pic_data, tsp_dis, num, 'Emulate')

    # plt.figure(figsize=(6, 6))
    k = 2
    nodes, dis = SOM_TSP(data, k * num, 1000, 0.5, 5, flex_size)
    plt.plot(data[:, 0], data[:, 1], 'r.', markersize=20)
    plotTSP(nodes, dis, num, 'SOM', k)