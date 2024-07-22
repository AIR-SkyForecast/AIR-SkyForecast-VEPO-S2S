import json
from sklearn.cluster import HDBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

# 计算轨迹点线性度的函数
def calculate_linearity(data):
    # 计算点之间的欧几里得距离
    distances = euclidean_distances(data)
    # 计算距离的平均值
    mean_distance = np.mean(distances)
    # 计算距离的标准差
    std_distance = np.std(distances)
    # 计算线性度，即标准差与平均距离的比值
    linearity = std_distance / mean_distance
    return linearity


def hdbscan():
    # 从文件加载数据
    with open('../../dataset_json/running_loc_merged.json') as f:
        load_dict = json.load(f)
    min_samples = 40
    # 存储聚类中心的字典
    cluster_centers_dict = {}
    xianxin = 0 #线性分布船数量
    wujvlei = 0  # 无聚类
    # 遍历每艘船
    for key, trajectory_points in tqdm(load_dict.items()):
        # 初始化聚类中心字典
        if trajectory_points:
            # 存储轨迹点的列表
            all_points = []
            # 存储聚类标签
            all_labels = []

            # 直接添加轨迹点到列表
            all_points.extend(trajectory_points)
            # 添加聚类标签的相同次数，以匹配轨迹点数量
            all_labels.extend([key] * len(trajectory_points))

            # 将轨迹点列表转换为NumPy数组
            data = np.array(all_points)

            # 检查数据点数量是否大于等于min_samples
            if len(data) >= min_samples:
                cluster_centers_dict[key] = []
                # 检查轨迹点的线性度
                linearity_threshold = 100  # 根据需要调整此阈值
                linearity = calculate_linearity(data)

                # 仅当轨迹点不呈明显直线分布时使用HDBSCAN进行聚类
                if linearity < linearity_threshold:
                    clusterer = HDBSCAN(min_cluster_size=40)
                    labels = clusterer.fit_predict(data)

                    # 获取聚类中心
                    cluster_centers = []
                    for cluster_label in np.unique(labels):
                        if cluster_label != -1:  # 排除噪声点
                            cluster_centers.append(np.mean(data[labels == cluster_label], axis=0))
                    cluster_centers = np.array(cluster_centers)

                    # 输出聚类中心
                    # print(f"Ship {key} 的聚类中心:", cluster_centers)

                    # 存储聚类中心到字典
                    cluster_centers_dict[key] = cluster_centers.tolist()

                    # 绘制聚类效果和中心
                    plt.figure(figsize=(10, 6))

                    # Check if there are cluster centers before plotting
                    if len(cluster_centers) > 0:
                        # Use np.where to create the boolean mask
                        mask = np.where(labels != -1)[0]
                        scatter = plt.scatter(data[mask, 0], data[mask, 1], c=labels[mask], cmap='rainbow', s=10, alpha=0.8)
                        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x',
                                    label='Cluster Centers')
                        # 添加颜色条和标签
                        plt.colorbar(scatter, label='Cluster Label')
                        plt.title(f'Trajectory Clustering - MMSI: {key}')
                        plt.legend()
                        plt.savefig("dbscan.pdf", dpi=300, format="pdf")
                        # 显示图形
                        plt.show()

                        a = input()

                    else:
                        wujvlei += 1
                        # Plot only the original trajectory points if there are no cluster centers
                        scatter = plt.scatter(data[:, 0], data[:, 1], c='gray', s=10, alpha=0.8)
                        # 添加颜色条和标签
                        plt.colorbar(scatter, label='Cluster Label')
                        plt.title(f'Trajectory Clustering - MMSI: {key}')
                        plt.legend()
                        plt.savefig("dbscan.pdf", dpi=300, format="pdf")
                        # 显示图形
                        plt.show()

                        a = input()

                else:
                    print(f"船 {key} 的轨迹点呈直线分布。跳过聚类.")
                    xianxin = xianxin + 1
                    # 对于呈直线分布的轨迹点，将空列表存储在字典中
                    cluster_centers_dict[key] = []

                    # 绘制原始数据分布
                    plt.figure(figsize=(10, 6))
                    scatter = plt.scatter(data[:, 0], data[:, 1], c='gray', s=10, alpha=0.8)
                    plt.title(f'Original Trajectory Distribution - MMSI: {key}')
                    # plt.show()

            else:
                # 输出信息，没有足够的数据点进行聚类
                print(f"船 {key} 的数据点不足以进行聚类.")

                # 对于数据点不足的船舶，将空列表存储在字典中
                # cluster_centers_dict[key] = []


    # 输出聚类中心的字典
    print("聚类中心字典:")
    print(cluster_centers_dict)

    print(len(load_dict))
    lll = []
    mmsi_lists = []
    i = 0
    for key in cluster_centers_dict.keys():
        l = len(cluster_centers_dict[key])
        lll.append(l)
        if cluster_centers_dict[key] != []:
            i += 1
            mmsi_lists.append(key)
    # 聚类蔟不为空的mmsi
    with open('../../running_loc/'  + 'mmsi.json', 'w', encoding='utf-8') as f:
        json.dump(mmsi_lists, f)

    # 将聚类蔟不为空的聚类中心写入json，为归一化做准备
    buweikong_cluster_centers = {}
    for key in cluster_centers_dict.keys():
        if key in mmsi_lists:
            buweikong_cluster_centers[key] = cluster_centers_dict[key]
    with open('../../running_loc/'  + 'buweikong_cluster_centers.json', 'w', encoding='utf-8') as f:
        json.dump(buweikong_cluster_centers, f)


    print('总船舶数： ', len(load_dict))
    print('活动轨迹大于',min_samples,'个轨迹点的船舶数：', len(cluster_centers_dict))
    print('活动轨迹大于',min_samples,'个但无聚类船舶数：', wujvlei)
    print('呈线性轨迹分布跳过聚类船舶数：', xianxin)
    print('聚类蔟不为空的船舶数：', i)
    print('最大聚类簇数： ', max(lll))
    print('最小聚类簇数： ', min(lll))
    print()
    print( len(cluster_centers_dict),'条船中',i ,'条船聚类不为空')


def normalization(data=None, min_=None, max_=None):
    data = float(data)
    new_a = (data - min_) / (max_ - min_)
    return new_a



def nor_hdbscan():
    mmsi_traj = {}
    length = [] # 聚类镞数
    # 获取经纬度最大值，最小值
    max_list = [-10000, -10000]  # lat,lon
    min_list = [100000, 110000]  # lat,lon
    # 从文件加载数据
    with open('../../running_loc/merged.json') as f:
        load_dict = json.load(f)
    # print(len(load_dict))
    with open(r'../../running_loc/mmsi.json', 'r') as f:
        mmsi_lists = json.load(f)
    # print(len(mmsi_lists))
    with open(r'../../running_loc/buweikong_cluster_centers.json', 'r') as f:
        buweikong_cluster_centers = json.load(f)
    # a = input()
    for key in load_dict.keys():
        if key in mmsi_lists:
            mmsi_traj[key] = load_dict[key]    # 包含有聚类蔟的船舶的航行轨迹点

    for mmsi, running_periods in mmsi_traj.items():
        for running_period in running_periods:
            curr_lat = running_period[0]
            if curr_lat > max_list[0]:
                max_list[0] = curr_lat
            elif curr_lat < min_list[0]:
                min_list[0] = curr_lat
            curr_lon = running_period[1]
            if curr_lon > max_list[1]:
                max_list[1] = curr_lon
            elif curr_lon < min_list[1]:
                min_list[1] =  curr_lon

    for mmsi, running_periods in buweikong_cluster_centers.items():
        for running_period in running_periods:
            nor_lat = normalization(running_period[0], min_list[0], max_list[0])
            nor_lon = normalization(running_period[1], min_list[1], max_list[1])
            running_period[0] = nor_lat
            running_period[1] = nor_lon


    for key in buweikong_cluster_centers.keys():
        length.append(len(buweikong_cluster_centers[key]))
    max_length = max(length)  #最大镞数
    print('最大镞数：', max_length)

    # 按照船只最大聚类镞数，补全所有船
    for key in buweikong_cluster_centers.keys():
        curr_len = len(buweikong_cluster_centers[key])
        for i in range(max_length-curr_len): #每条船后面要补上多少个[0.0,0.0]聚类蔟
            buweikong_cluster_centers[key].append([0.0, 0.0])

    with open('../../running_loc/'  + 'buquan_cluster_centers.json', 'w', encoding='utf-8') as f:
        json.dump(buweikong_cluster_centers, f)

    # print(buweikong_cluster_centers)
    # print(len(buweikong_cluster_centers))
    # print(max_length)

# 9012条船轴  6194条聚类不为空
hdbscan()
# nor_hdbscan()







