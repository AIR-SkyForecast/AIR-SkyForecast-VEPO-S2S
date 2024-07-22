import json
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
# ******************* 相对路径问题。
import sys
sys.path.append("..")   # ***起作用
sys.path.append("./vepo_lstm")  # 可删除
import os
print("***************", os.getcwd())
# *******************
from vepo_gru.vepo_gru_preprocess.vepo_gru_conf import batch_size, len_train, len_label
import torch.nn as nn
import numpy as np
import random


def gen_dataloader():
    train_path=r'../dataset_json/train.json'
    val_path=r'../dataset_json/val.json'
    test_path=r'../dataset_json/test.json'
    with open(train_path, 'r') as f:
        train = json.load(f)
    with open(val_path, 'r') as f:
        val = json.load(f)
    with open(test_path, 'r') as f:
        test = json.load(f)
    train = trajectory_cut(train)   #0.8
    val = trajectory_cut(val)   #0.1
    test = trajectory_cut(test) #0.1

    max_min_path=r'../dataset_json/max_min.json'
    with open(max_min_path,'r') as f:
        max_min = json.load(f)
        max_ = max_min[0]
        min_ = max_min[1]

    train_data = get_dataloader(train, batch_size)
    val_data = get_dataloader(val, batch_size)
    test_data = get_dataloader(test, batch_size)

    return train_data, val_data, test_data, max_, min_


 # 滑动窗口进行轨迹切分
def trajectory_cut(data):
    list_10=[]
    for running_period in data:
        left = 0
        right = 15
        while right<len(running_period):
            train_seq = running_period[left:right]
            left +=3   # 改大一些，数据量就小了
            right +=3
            list_10.append(train_seq)
    return list_10


def get_dataloader(data, batch_size, shuffle=False):
    dataset = Mydataset(data)
    dataloder = DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                            num_workers = 0, drop_last = True)
    return dataloder

#

class Mydataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        train_data = self.data[item][0:len_train]
        label_data = self.data[item][len_train:len_train+len_label]
        mmsi_id = self.data[item][0][-1]

        return torch.Tensor(train_data),torch.Tensor(label_data),mmsi_id



class Input_Module(nn.Module):
    """
    Embedding and concatenating the spatiotemporal semantics， trajectory points, and  driving status
    """
    embed_dim = [("month", 13, 2), ("day", 32, 2), ("hour", 25, 2), ("type", 101, 8)]

    def __init__(self):
        super(Input_Module, self).__init__()
        # 对时间和
        for name, num_embeddings, embedding_dim in Input_Module.embed_dim:
            self.add_module(name + '_embed', nn.Embedding(num_embeddings, embedding_dim))  # ！！！！

    def forward(self, train):
        train_loc_semantic, train_property_semantic, train_type_semantic, train_time_semantic = Matrix_slice(train)

        attr = {
            "month": train_time_semantic[:, :, 0],
            "day": train_time_semantic[:, :, 1],
            "hour": train_time_semantic[:, :, 2],
            "type": train_type_semantic
        }
        time_semantic = []
        for name, num_embeddings, embedding_dim in Input_Module.embed_dim:
            #   gerattr()返回一个对象属性值
            embed = getattr(self, name + '_embed')
            _attr = embed(attr[name])
            time_semantic.append(_attr)
        # time_type_semantic[128,10,14]
        # time_type_semantic = torch.cat(time_semantic, dim=2)  # 3. 时间和类型语义拼接

        train_type_semantic = time_semantic[-1]
        # #   将船舶类型划分为100个类别，嵌入到8纬向量中
        # type_embed = torch.nn.Embedding(101, 8)
        # #   label_data[2]为船舶类型的语义信息[128,5]  type_semantic [128,5,8]
        # type_semantic = type_embed(label_data[2])  # 类型语义tensor，暂不使用，后续使用


        return train_type_semantic



def Matrix_slice(data=None):
    """传入三维tensor进行切片操作"""
    #   lat lon
    loc_semantic = data[:, :, 1:3]
    #   角度 速度 航行距离
    property_semantic = data[:, :, 3:6]

    #   船类型
    type_semantic = data[:, :, 6].long()
    #   月 日 小时
    time_semantic = data[:, :, 7:10].long()
    return loc_semantic, property_semantic, type_semantic, time_semantic
