import copy
from vepo_gru.vepo_gru_preprocess.vepo_gru_models import LSTM, GRU, RMSELoss, MAE, ADELoss, FDELoss, BiLSTM, Seq2Seq\
    , Feature_Encoder,  reverse_normalize
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import json
import os
from matplotlib import pyplot as plt
import random  # 添加这一行



# 不设置会报错
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
path = r'../models/vepo_gru.pkl'
draw_save_path = r"vepo_gru"



def load_models(args):
    input_size, Feature_input_size, hidden_size = args.input_size, args.Feature_input_size,args.hidden_size
    num_layers, decoder_num_layers, output_size  = args.num_layers, args.decoder_num_layers, args.output_size,
    batch_size, embedding_dim =  args.batch_size, args.embedding_dim
    if args.flag in ['Bilstm']:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size).to(device)
    else:
        model = Seq2Seq(input_size, Feature_input_size, hidden_size,
                        num_layers, output_size, batch_size, embedding_dim, decoder_num_layers).to(device)

    return model


def get_val_loss(model, val, five_feature, buquan_centers, nor_running_time,  nor_parking_time, args):
    model.eval()
    loss_function = RMSELoss().to(device)
    val_loss = []
    # print('validing.....')

    for step, (seq, target, mmsi_id) in enumerate(val):
        parking_time_feature_list = []
        running_time_feature_list = []
        running_loc_feature_list = []
        five_feature_lists = []

        # 依次顺序装填seq中每条mmsi对应的feature
        for i in mmsi_id:
            i = str(int(i))
            five_feature_lists.append(five_feature[i][1:6])
            parking_time_feature_list.append(nor_parking_time[i])
            running_time_feature_list.append(nor_running_time[i])
            running_loc_feature_list.append(buquan_centers[i])

        # feature_tensor[128, 5] 类型、长、宽、 吨位、吃水
        feature_tensor = np.array([np.array(feature).astype(np.float32) for feature in five_feature_lists])
        feature_tensor = torch.tensor(feature_tensor)

        parking_time_feature_list = np.array(
            [np.array(feature).astype(np.float32) for feature in parking_time_feature_list])
        running_time_feature_list = np.array(
            [np.array(feature).astype(np.float32) for feature in running_time_feature_list])
        running_loc_feature_list = np.array(
            [np.array(feature).astype(np.float32) for feature in running_loc_feature_list])

        # 将numpy转成tensor
        parking_time_feature_list_tensor = torch.tensor(parking_time_feature_list)
        parking_time_feature_list_tensor = parking_time_feature_list_tensor.to(device)

        running_time_feature_list_tensor = torch.tensor(running_time_feature_list)
        running_time_feature_list_tensor = running_time_feature_list_tensor.to(device)

        running_loc_feature_list_tensor = torch.tensor(running_loc_feature_list)
        running_loc_feature_list_tensor = running_loc_feature_list_tensor.to(device)

        with torch.no_grad():
            seq = seq[:, :, 1:6]
            target = target[:, :, 1:3]
            seq = seq.to(device)
            target = target.to(device)

            y_pred, kl_loss = model(seq, feature_tensor, parking_time_feature_list_tensor, running_time_feature_list_tensor,
                                    running_loc_feature_list_tensor, target, False)
            loss = loss_function(y_pred, target)
            loss = loss + kl_loss
            val_loss.append(loss.item())

    # print('val_loss:{}'.format(np.mean(val_loss)))
    return np.mean(val_loss)




def  draw(train_loss_list, val_loss_list, test_loss_list, save_path):

    y_train_loss = train_loss_list  # loss值，即y轴
    x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴

    y_val_loss = val_loss_list  # loss值，即y轴
    x_val_loss = range(len(y_val_loss))  # loss的数量，即x轴

    y_test_loss = test_loss_list  # loss值，即y轴
    x_test_loss = range(len(y_test_loss))  # loss的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    plt.ylabel('loss')  # y轴标签
    plt.ylim([0, 0.05])
    plt.yticks(np.arange(0, 0.05, 0.005))

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss", )
    plt.plot(x_val_loss, y_val_loss, linewidth=1, linestyle="solid",
             label="val loss")
    plt.plot(x_test_loss, y_test_loss, linewidth=1, linestyle="solid",
             label="test loss")

    plt.xticks(range(0, len(x_train_loss)+1, 10))
    plt.legend()
    plt.title('s2s_gru_Loss curve')

    if save_path:
        plt.savefig(save_path)

    plt.show()



def train(args, Dtr, Val, Dte):
    # global val_loss, test_loss
    model = load_models(args)
    loss_function = RMSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    schduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    min_epochs = -1
    val_loss = 0
    test_loss = 0
    min_val_loss = 30
    best_model = None
    draw_train_loss = []
    draw_val_loss = []
    draw_test_loss = []
    # 五类浅层特征
    with open(r'../dataset_json/nor_mmsi_list.json', 'r') as f:
        five_feature = json.load(f)
        list = five_feature.keys() #6194

    # 聚类位置
    with open(r'../dataset_json/buquan_cluster_centers.json', 'r') as f:
        buquan_centers = json.load(f)
        list = buquan_centers.keys()   #6194

    # 活动时间
    with open(r'../dataset_json/nor_running_time.json', 'r') as f:
        nor_running_time = json.load(f)
        list = nor_running_time.keys()  #

    # 停泊时间
    with open(r'../dataset_json/nor_parking_time.json', 'r') as f:
        nor_parking_time = json.load(f)
        list = nor_parking_time.keys()  #6194
        # print(len(list))

    for epoch in range(40):
        train_loss = []
        for step, (seq, label, mmsi_id) in enumerate(Dtr):
            five_feature_lists = []
            parking_time_feature_list = []
            running_time_feature_list = []
            running_loc_feature_list = []
            model.train()   # **************
            # 依次顺序装填seq中每条mmsi对应的feature
            for i in mmsi_id:
                i = str(int(i))
                five_feature_lists.append(five_feature[i][1:6])
                parking_time_feature_list.append(nor_parking_time[i])
                running_time_feature_list.append(nor_running_time[i])
                running_loc_feature_list.append(buquan_centers[i])

            # feature_tensor[128, 5] 类型、长、宽、 吨位、吃水
            feature_tensor = np.array([np.array(feature).astype(np.float32) for feature in  five_feature_lists])
            feature_tensor = torch.tensor(feature_tensor)

            parking_time_feature_list = np.array([np.array(feature).astype(np.float32) for feature in parking_time_feature_list])
            running_time_feature_list = np.array([np.array(feature).astype(np.float32) for feature in running_time_feature_list])
            running_loc_feature_list = np.array([np.array(feature).astype(np.float32) for feature in running_loc_feature_list])
            # 将numpy转成tensor
            parking_time_feature_list_tensor = torch.tensor(parking_time_feature_list)
            parking_time_feature_list_tensor = parking_time_feature_list_tensor.to(device)

            running_time_feature_list_tensor = torch.tensor(running_time_feature_list)
            running_time_feature_list_tensor = running_time_feature_list_tensor.to(device)

            running_loc_feature_list_tensor = torch.tensor(running_loc_feature_list)
            running_loc_feature_list_tensor = running_loc_feature_list_tensor.to(device)

            # 输入[128, 10, 5]、标签[128, 5, 2]
            seq = seq[:,:,1:6]
            label = label[:,:,1:3]
            seq = seq.to(device)
            label = label.to(device)

            y_pred, kl_loss = model(seq, feature_tensor, parking_time_feature_list_tensor, running_time_feature_list_tensor,
                                    running_loc_feature_list_tensor, label)     # *************
            loss = loss_function(y_pred, label)
            loss = loss + kl_loss
            train_loss.append(loss.item())
            # 反向传播和优化
            optimizer.zero_grad()           # **************
            loss.backward()
            optimizer.step()
            step += 1
            #   每500step做一次val和test,更新学习率
            if step % 10 == 0:
                print('epoch:{}, step :{}, train_loss:{}'.format(epoch, step, np.mean(train_loss)))
            if step % 500 == 0:
                schduler.step()

        val_loss = get_val_loss(model, Val, five_feature, buquan_centers, nor_running_time, nor_parking_time, args)
        print('epoch:{}, step :{}, train_loss:{} val_loss:{}'.format(epoch, step, np.mean(train_loss), val_loss))

        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)


    state = {'models':best_model.state_dict()}
    print('best_model:{} min_val_loss:{}'.format(best_model, min_val_loss))


    torch.save(state, path)





def test1(args, Dte):
    test_loss = []
    mae_loss = []
    ade_loss = []
    fde_loss = []
    loss_dict_rmse = {}
    loss_t_dict_rmse = {}
    loss_dict_mae = {}
    loss_t_dict_mae = {}
    loss_dict_ade = {}
    loss_t_dict_ade = {}
    loss_dict_fde = {}
    # 五类浅层特征
    with open(r'../dataset_json/nor_mmsi_list.json', 'r') as f:
        five_feature = json.load(f)
        list = five_feature.keys()

    # 聚类位置
    with open(r'../dataset_json/buquan_cluster_centers.json', 'r') as f:
        buquan_centers = json.load(f)
        list = buquan_centers.keys()  # 6194

    # 活动时间
    with open(r'../dataset_json/nor_running_time.json', 'r') as f:
        nor_running_time = json.load(f)
        list = nor_running_time.keys()  # 6194

    # 停泊时间
    with open(r'../dataset_json/nor_parking_time.json', 'r') as f:
        nor_parking_time = json.load(f)
        list = nor_parking_time.keys()  # 6194
        # print(len(list))

    #   加载模型
    model = load_models(args)
    model.load_state_dict(torch.load(path, map_location='cpu')['models'])
    #   评价指标
    loss_function = RMSELoss().to(device)
    mae_loss_function = MAE().to(device)
    ade_loss_function = ADELoss().to(device)
    fde_loss_function = FDELoss().to(device)

    model.eval()
    print('predicting....')
    for step, (seq, label, mmsi_id) in enumerate(Dte):
        seq_len = label.shape[1]
        parking_time_feature_list = []
        running_time_feature_list = []
        running_loc_feature_list = []
        five_feature_lists = []
        # 依次顺序装填seq中每条mmsi对应的feature
        for i in mmsi_id:
            i = str(int(i))
            five_feature_lists.append(five_feature[i][1:6])
            parking_time_feature_list.append(nor_parking_time[i])
            running_time_feature_list.append(nor_running_time[i])
            running_loc_feature_list.append(buquan_centers[i])

        # feature_tensor[128, 5] 类型、长、宽、 吨位、吃水
        feature_tensor = np.array([np.array(feature).astype(np.float32) for feature in five_feature_lists])
        feature_tensor = torch.tensor(feature_tensor)
        parking_time_feature_list = np.array(
            [np.array(feature).astype(np.float32) for feature in parking_time_feature_list])
        running_time_feature_list = np.array(
            [np.array(feature).astype(np.float32) for feature in running_time_feature_list])
        running_loc_feature_list = np.array(
            [np.array(feature).astype(np.float32) for feature in running_loc_feature_list])

        # 将numpy转成tensor
        parking_time_feature_list_tensor = torch.tensor(parking_time_feature_list)
        parking_time_feature_list_tensor = parking_time_feature_list_tensor.to(device)

        running_time_feature_list_tensor = torch.tensor(running_time_feature_list)
        running_time_feature_list_tensor = running_time_feature_list_tensor.to(device)

        running_loc_feature_list_tensor = torch.tensor(running_loc_feature_list)
        running_loc_feature_list_tensor = running_loc_feature_list_tensor.to(device)

        with torch.no_grad():
            seq = seq[:, :, 1:6]
            label = label[:, :, 1:3]
            seq = seq.to(device)
            label = label.to(device)

            y_pred, kl_loss = model(seq, feature_tensor, parking_time_feature_list_tensor, running_time_feature_list_tensor,
                                    running_loc_feature_list_tensor, label, False)

            loss = loss_function(y_pred, label)
            loss = loss + kl_loss
            test_loss.append(loss.item())

            mae = mae_loss_function(y_pred, label)
            mae = mae + kl_loss
            mae_loss.append(mae.item())

            ade = ade_loss_function(y_pred, label)
            ade = ade + kl_loss
            ade_loss.append(ade.item())

            fde = fde_loss_function(y_pred[:-1:], label[:-1:])
            fde = fde + kl_loss
            fde_loss.append(fde.item())


            for t in range(seq_len):
                if t not in loss_dict_rmse.keys():
                    loss_dict_rmse[t] = []
                loss_dict_rmse[t].append((loss_function(y_pred[:, t, :], label[:, t, :])+ kl_loss).item())
                if t not in loss_t_dict_rmse.keys():
                    loss_t_dict_rmse[t] = []
                loss_t_dict_rmse[t].append((loss_function(y_pred[:, :t + 1, :], label[:, :t + 1, :])+ kl_loss).item())

            for t in range(seq_len):
                if t not in loss_dict_mae.keys():
                    loss_dict_mae[t] = []
                loss_dict_mae[t].append((mae_loss_function(y_pred[:, t, :], label[:, t, :])+ kl_loss).item())
                if t not in loss_t_dict_mae.keys():
                    loss_t_dict_mae[t] = []
                loss_t_dict_mae[t].append((mae_loss_function(y_pred[:, :t + 1, :], label[:, :t + 1, :])+ kl_loss).item())

            for t in range(seq_len):
                if t not in loss_t_dict_ade.keys():
                    loss_t_dict_ade[t] = []
                loss_t_dict_ade[t].append((ade_loss_function(y_pred[:, :t + 1, :], label[:, :t + 1, :])+ kl_loss).item())

            for t in range(seq_len):
                if t not in loss_dict_fde.keys():
                    loss_dict_fde[t] = []
                loss_dict_fde[t].append((fde_loss_function(y_pred[:, t, :], label[:, t, :])+ kl_loss).item())

    for i, loss_ in loss_dict_rmse.items():
        print('predict the {} point mean loss is {}'.format(i + 1, np.mean(loss_)))
    print('')
    for i, _loss in loss_t_dict_rmse.items():
        print('predict the first {} points` mean loss is {}'.format(i + 1, np.mean(_loss)))
    print('')

    for i, loss_ in loss_dict_mae.items():
        print('mae predict the {} point mean loss is {}'.format(i + 1, np.mean(loss_)))
    print('')
    for i, _loss in loss_t_dict_mae.items():
        print('mae test predict the first {} points` mean loss is {}'.format(i + 1, np.mean(_loss)))
    print('')

    for i, _loss in loss_t_dict_ade.items():
        print('ade test predict the first {} points` mean loss is {}'.format(i + 1, np.mean(_loss)))
    print('')

    for i, loss_ in loss_dict_fde.items():
        print('fde predict the {} point mean loss is {}'.format(i + 1, np.mean(loss_)))
    print('')


    print('test mean rmse is {}, mae is {}, ade is {} '.format(np.mean(test_loss), np.mean(mae_loss)
                                                               ,np.mean(ade_loss) ))




