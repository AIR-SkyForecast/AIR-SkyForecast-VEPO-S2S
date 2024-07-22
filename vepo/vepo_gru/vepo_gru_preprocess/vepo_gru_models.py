import torch
from torch import nn
import os
import math
import numpy as np
import torch.nn.functional as F
from vepo_gru.vepo_gru_preprocess.vepo_gru_conf import len_label


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x ,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x,y))
        return loss


# 计算MAPE指标的函数
def MAPE(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class MAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        criterion = nn.L1Loss()
        loss = criterion(x, y)
        return loss


class ADELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_trajectory, true_trajectory):
        errors = torch.sqrt(torch.sum((predicted_trajectory - true_trajectory)**2, dim=-1))
        ade = torch.mean(errors)
        return ade

class FDELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_final_position, true_final_position):
        fde = torch.sqrt(torch.sum((predicted_final_position - true_final_position)**2, dim=-1))
        return torch.mean(fde)  # or torch.sum(fde) for the sum of distances


def reverse_normalize(normalized_lat, normalized_lon, min_lat, min_lon, max_lat, max_lon):
    # 反标准化（反归一化）
    original_lat = normalized_lat * (max_lat - min_lat) + min_lat
    original_lon = normalized_lon * (max_lon - min_lon) + min_lon
    return original_lat.tolist(), original_lon.tolist()

def kl_divergence(P, Q, epsilon=1e-10):
    # 初始化 KL 散度列表

    total_kl_loss = 0
    # 遍历每组概率分布
    for p, q in zip(P, Q):
        # 平滑处理，避免分母为零
        p = torch.clamp(p, min=epsilon)
        q = torch.clamp(q, min=epsilon)

        # 计算每个样本的 KL 散度
        elementwise_kl = p * torch.log(p / q)

        # 对分布所有事件取和
        samplewise_sum = torch.sum(elementwise_kl)

        total_kl_loss += samplewise_sum

    return total_kl_loss


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_direction = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size,seq_len = input_seq[0], input_seq[1]
        h0 = torch.randn(self.num_direction * self.n_layers, self.batch_size, self.hidden_size).to(device)
        c0 = torch.randn(self.num_direction * self.n_layers, self.batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq,(h0,c0))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred




class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size * 2
        self.n_layers = n_layers * 2
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_direction = 1
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq[0], input_seq[1]
        h0 = torch.randn(self.num_direction * self.n_layers, self.batch_size, self.hidden_size).to(device)
        # c0 = torch.randn(self.num_direction * self.n_layers, self.batch_size, self.hidden_size).to(device)
        output, _ = self.gru(input_seq, h0)
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred





class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        input_seq = input_seq.view(self.batch_size, seq_len, self.input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
       #  output(, , 0,)表示前隐藏状态  output(, , 1,)表示后隐藏状态
        output = output.contiguous().view(self.batch_size, seq_len, self.num_directions, self.hidden_size)
        # 将前后隐藏状态用均值方式合并，这时output维度变为(batch_size, seq_len,  hidden_size)
        output = torch.mean(output, dim=2)
        pred = self.linear(output)
        # print('pred=', pred.shape)
        pred = pred[:, -1, :]
        return pred



class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(GRUWithAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.number_directions = 1
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.attention = nn.Linear(hidden_size ,1)
        self.fc = nn.Linear(hidden_size , 2)

    def forward(self, x):
        h_0 = torch.randn(self.num_layers * self.number_directions, self.batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_layers * self.number_directions, self.batch_size, self.hidden_size).to(device)
        gru_out, _ = self.gru(x, h_0)
        # 注意力机制
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)
        attention_output = torch.sum(attention_weights * gru_out, dim=1)
        Ct = self.fc(attention_output)
        return Ct ,attention_output


class Five_FeatureGRUWithEmbedding(nn.Module):
    def __init__(self, hidden_size, num_layers, batch_size, embedding_dim):
        super(Five_FeatureGRUWithEmbedding, self).__init__()
        self.input_size = embedding_dim
        self.num_directions = 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(1, embedding_dim, dtype=torch.float)
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(self.hidden_size, self.batch_size)


    def forward(self, x):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        x = x.unsqueeze(2).to(device) #128 4 1
        embedding = self.embedding(x) #128 4 36
        output, _ = self.gru(embedding, h_0) #128 4 hidden_size
        output = self.fc(output)

        # 全连接层
        # output = self.fc(output)

        return output



class Type_embedding(nn.Module):
    def __init__(self, hidden_size, num_layers, batch_size, embedding_dim):
        super(Type_embedding, self).__init__()
        self.input_size = embedding_dim
        self.num_directions = 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(1, embedding_dim, dtype=torch.float)
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(self.hidden_size, self.batch_size)

    def forward(self, x):
        # with torch.no_grad():
            h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
            # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
            # x = x.unsqueeze(1)# 128 1
            x = x.unsqueeze(2).to(device) #128 1 1
            embedding = self.embedding(x) #128 1 36
            output, _ = self.gru(embedding, h_0) #128 1 hidden_size
            output = self.fc(output)

            # 全连接层
            # output = self.fc(output)

            return output



class Label_GRUWithEmbedding(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, batch_size):
        super(Label_GRUWithEmbedding, self).__init__()
        self.input_size = output_size
        self.num_directions = 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(self.hidden_size , 2)

    def forward(self, label):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        gru_out, _ = self.gru(label, h_0)
        # 全连接层
        output = self.fc(gru_out)
        return output


class Time_Feature_GRU(nn.Module):
    def __init__(self, hidden_size, num_layers, batch_size, embedding_dim):
        super(Time_Feature_GRU, self).__init__()
        self.input_size = 24
        self.num_directions = 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size , self.num_layers, batch_first=True, bidirectional=False)
        self.embedding = nn.Linear(self.input_size, self.embedding_dim, dtype=torch.float)
        self.linear = nn.Linear(self.hidden_size, self.batch_size)

    def forward(self, feature_tensor):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size ).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size * 2).to(device)
        feature_tensor = self.embedding(feature_tensor)
        output, _ = self.gru(feature_tensor, h_0)
        output = self.linear(output)
        return output

class Loc_Feature_GRU(nn.Module):
    def __init__(self, hidden_size, num_layers, batch_size, embedding_dim):
        super(Loc_Feature_GRU, self).__init__()
        self.input_size = 114
        self.num_directions = 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(self.input_size, self.embedding_dim, dtype=torch.float)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size , self.num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.batch_size)

    def forward(self, feature_tensor):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size ).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size * 2).to(device)
        feature_tensor = feature_tensor.unsqueeze(1)
        feature_tensor = self.embedding(feature_tensor)
        output, _ = self.gru(feature_tensor, h_0)
        output = self.linear(output)
        return output



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        # self.gru_attention = GRUWithAttention(self.input_size, self.hidden_size, self.num_layers, self.batch_size)
        self.fc = nn.Linear(self.hidden_size , 2)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size ).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size ).to(device)
        output, h = self.gru(input_seq, h_0)
        H = output
        output = output[:, -1, :]
        output = self.fc(output)
        # Ct ,attention_output = self.gru_attention(input_seq)
        return H, output , h


class Feature_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, embedding_dim):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.Type_Embedding = Type_embedding(self.hidden_size, self.num_layers, self.batch_size, self.embedding_dim)
        self.Five_FeatureGRUWithEmbedding = Five_FeatureGRUWithEmbedding(self.hidden_size, self.num_layers, self.batch_size, self.embedding_dim)
        self.Time_Feature_GRU = Time_Feature_GRU(self.hidden_size, self.num_layers, self.batch_size, self.embedding_dim)
        self.Loc_Feature_GRU = Loc_Feature_GRU( self.hidden_size, self.num_layers,self.batch_size, self.embedding_dim)
        self.linear = nn.Linear(self.hidden_size, self.batch_size)
    def forward(self, five_feature_tensor, parking_time_feature_list_tensor, running_time_feature_list_tensor,
                running_loc_feature_list_tensor):
        type_tensor = five_feature_tensor[:,0]
        type_tensor = type_tensor.unsqueeze(1)
        type_tensor = type_tensor.to(device)
        type_tensor = self.Type_Embedding(type_tensor)
        parking_time_feature_list_tensor = parking_time_feature_list_tensor.unsqueeze(1)
        running_time_feature_list_tensor = running_time_feature_list_tensor.unsqueeze(1)
        four_feature_tensor = self.Five_FeatureGRUWithEmbedding(five_feature_tensor[:,1:])
        running_loc_feature_list_tensor =  running_loc_feature_list_tensor.reshape(self.batch_size, -1)
        time_tensor = torch.cat((parking_time_feature_list_tensor, running_time_feature_list_tensor), dim=1)
        time_output = self.Time_Feature_GRU(time_tensor)
        loc_output = self.Loc_Feature_GRU(running_loc_feature_list_tensor)
        output = torch.cat((type_tensor, four_feature_tensor, time_output, loc_output), dim=1)
        return output


class Label_Encoder(nn.Module):
    # Label_Encoder与Encoder参数不共享
    def __init__(self, output_size, label_hidden_size, label_num_layers, batch_size):
        super().__init__()
        self.input_size = output_size
        self.label_hidden_size = label_hidden_size
        self.label_num_layers = label_num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.gru_embedding = Label_GRUWithEmbedding(self.output_size, self.label_hidden_size, self.label_num_layers, self.batch_size)

    def forward(self, input_seq):
        output = self.gru_embedding(input_seq)
        return output


class Prior_Posterior(nn.Module):
    def __init__(self):
        super(Prior_Posterior, self).__init__()
    # output[128, 2]  feature_out[128, 3, 128]  Encoded_label[128,5,2]
    def forward(self,batch_size, output, feature_output, Encoded_label):
        # 先验分子、分母、值
        prior_numerator = 0.0
        prior_denominator = []
        prior = []
        for i in range(0,8):
            each_feature_output = feature_output[:, i, :]
            prior_numerator = torch.matmul(each_feature_output, output)
            after_exp = torch.exp(prior_numerator)
            after_exp = after_exp.unsqueeze(1)
            prior_denominator.append(after_exp)

        for j in range(0,8):
           prior.append(prior_denominator[j]/sum(prior_denominator))
        # 将5个（128，2）个先验值合并成一个（128，5，2）
        prior = torch.cat(prior, dim=1)


        # 后验分子、分母、值
        posterior_numerator = 0.0
        posterior_denominator = []
        posterior = []
        output = output.unsqueeze(1)
        cat_output_label = torch.cat((output, Encoded_label), dim=1)
        linear_layer = nn.Linear(12, 2).to(device)
        # 将张量展平成 (batch_size 11*2)
        tensor_flattened = cat_output_label.view(batch_size, -1)
        # 通过线性层得到输出，然后将输出形状变为 (batch_size, 2)
        cat_output_label = linear_layer(tensor_flattened).view(batch_size, 2)

        for i in range(0,8):
            each_feature_output = feature_output[:, i, :]
            posterior_numerator = torch.matmul(each_feature_output, cat_output_label)
            after_exp = torch.exp(posterior_numerator)
            after_exp = after_exp.unsqueeze(1)
            posterior_denominator.append(after_exp)

        for j in range(0,8):
           posterior.append(posterior_denominator[j]/sum(posterior_denominator))
        ## 将5个（128，2）个后验值合并成一个（128，5，2）
        posterior = torch.cat(posterior, dim=1)
        kl_loss = kl_divergence(posterior, prior, epsilon=1e-10)


        return prior, kl_loss


class Knowledge_Fuse(nn.Module):
    def __init__(self,input_size, num_layers, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False).to(device)
        self.linear = nn.Linear(self.hidden_size * self.num_direction, 2).to(device)
    def forward(self,H, prior):
        batch_size = prior.shape[0]
        prior_poster = prior.view(self.num_layers, batch_size, self.hidden_size)
        h = prior_poster.to(device)
        c = h.clone().to(device)
        # self.lstm.to(H.device)
        output, (h, c) = self.lstm(H, (h, c))
        output = output[:, -1, :]
        output = self.linear(output)
        return output


class  Decoder_base(nn.Module):
    def __init__(self, output_size, num_layers, hidden_size, batch_size):
        super().__init__()
        self.input_size = output_size
        self.num_layers = num_layers
        self.num_direction = 1
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, 2)
        # self._linear = nn.Linear(self.input_size * 2, 2)


    def forward(self, input_seq, h):
        batch_size = input_seq.shape[0]
        input_size = input_seq.shape[1]
        input_seq = input_seq.view(batch_size, 1, input_size)
        output, h = self.gru(input_seq, h)
        pred = self.linear(output)  # pred(batch_size, 1, output_size)
        pred = pred[:, -1, :]
        return pred, h


class Decoder_FFAG(nn.Module):
    def __init__(self, output_size, num_layers, hidden_size, batch_size):
        super().__init__()
        self.input_size = output_size * 3
        self.hidden_size = hidden_size
        self.num_direction = 1
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size , 2)
    def forward(self,input_seq, kf, ctk, h):
        batch_size = input_seq.shape[0]
        input_seq = input_seq.view(batch_size, 1, -1)
        kf = kf.view(batch_size, 1, -1)
        ctk = ctk.view(batch_size, 1, -1)
        input_seq = torch.cat((input_seq, ctk, kf), dim=2)
        output, h  = self.gru(input_seq, h)
        pred = self.linear(output)  # pred(batch_size, 1, output_size)
        pred = pred[:, -1, :]
        return pred, h


class FusionGate(nn.Module):

    def __init__(self, input_size):
        super(FusionGate, self).__init__()

        # 权重矩阵
        self.W_y = nn.Linear(input_size, input_size).to(device)
        self.W_k = nn.Linear(input_size, input_size).to(device)
        self.Wz = nn.Linear(2 * input_size, input_size).to(device)

        # 激活函数
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, Sc, Sk):
        # 计算 FusionGate 输出
        Sc = Sc.to(Sk.device)
        Ot_input = torch.cat((self.tanh(self.W_y(Sc)), self.tanh(self.W_k(Sk))), dim=1)
        Ot = self.sigmoid(self.Wz(Ot_input))

        # 计算最终输出
        St = Ot * Sc + (1 - Ot) * Sk

        return St




class Seq2Seq(nn.Module):
    def __init__(self, input_size, Feature_input_size,
                 hidden_size, num_layers, output_size, batch_size , embedding_dim, decoder_num_layers):
        super().__init__()
        self.output_size = output_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.num_direction = 2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(input_size, hidden_size, num_layers, output_size, batch_size)
        self.Encoder = Encoder(input_size, hidden_size, num_layers, batch_size)
        self.Feature_Encoder = Feature_Encoder(Feature_input_size, hidden_size, num_layers, batch_size, embedding_dim)
        self.Label_Encoder = Label_Encoder(output_size, hidden_size, num_layers, batch_size)
        self.Knowledge_Fuse = Knowledge_Fuse(input_size, num_layers, hidden_size)
        self.Decoder_base = Decoder_base(output_size, decoder_num_layers, hidden_size, batch_size)
        self.Decoder_FFAG = Decoder_FFAG(output_size, decoder_num_layers, hidden_size, batch_size)

    def forward(self,input_seq, five_feature_tensor, parking_time_feature_list_tensor, running_time_feature_list_tensor,
                running_loc_feature_list_tensor, label, Train=True):
        batch_size = input_seq.shape[0]
        #   Encoder
        H, encoder_output, encoder_h = self.Encoder(input_seq)
        # feature_output[128,3,128]
        feature_output = self.Feature_Encoder(five_feature_tensor, parking_time_feature_list_tensor, running_time_feature_list_tensor,
                running_loc_feature_list_tensor)
        # 将label[128,10，2]经过标签编码变成[128,2]，提取最后一个output,代表预测轨迹的表示
        Encoded_label = self.Label_Encoder(label)

        # 先验后验
        prior_posterior = Prior_Posterior()
        prior, kl_loss = prior_posterior(batch_size, encoder_output, feature_output, Encoded_label)

        # Decoder
        pred_num = len_label
        outputs = torch.zeros(batch_size, pred_num, self.output_size).to(device)

        knowledge_fused = Knowledge_Fuse(input_size=self.hidden_size, hidden_size=8, num_layers=2)
        ctk = knowledge_fused(H, prior)
        kf = prior.clone()
        kf = kf.view(batch_size, -1)
        linear = nn.Linear(kf.shape[1], 2).to(device)
        kf = linear(kf)

        if Train:
            for t in range(pred_num):
                input = input_seq[:, -1, :2] if t == 0 else label[:, t - 1, :]
                if t == 0:         # 将语义向量C作为Decoder的h,c输入
                    base_output, h = self.Decoder_base(input, encoder_h)
                    #   FFAG
                    FFAG_output, h_ffag = self.Decoder_FFAG(input, kf, ctk, encoder_h)
                    fusiongate = FusionGate(input_size=2)
                    output = fusiongate(base_output, FFAG_output)
                    outputs[:, t, :] = output

                else:
                    base_output, h = self.Decoder_base(input, h)
                    FFAG_output, h_ffag = self.Decoder_FFAG(input, kf, ctk, h_ffag)
                    fusiongate = FusionGate(input_size=2)
                    output = fusiongate(base_output, FFAG_output)
                    outputs[:, t, :] = output

        else:
            # 第一个值用训练的最后一个值，后面每次都用预测产生的值
            base_output = input_seq[:, -1, :2]
            FFAG_output = input_seq[:, -1, :2]
            for t in range(pred_num):

                if t == 0:  # 将语义向量C作为Decoder的h,c输入
                    base_output, h = self.Decoder_base(base_output, encoder_h)
                    #   FFAG
                    FFAG_output, h_ffag = self.Decoder_FFAG(FFAG_output, kf, ctk, encoder_h)
                    fusiongate = FusionGate(input_size=2)
                    output = fusiongate(base_output, FFAG_output)
                    outputs[:, t, :] = output

                else:
                    base_output, h = self.Decoder_base(base_output, h)
                    FFAG_output, h_ffag = self.Decoder_FFAG(FFAG_output, kf, ctk, h_ffag)
                    fusiongate = FusionGate(input_size=2)
                    output = fusiongate(base_output, FFAG_output)
                    outputs[:, t, :] = output

        return outputs, kl_loss





