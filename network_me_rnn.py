import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from thop import profile
from thop import clever_format

# class LFPointTransformer(nn.Module):
#     def __init__(self):
#         super(LFPointTransformer, self).__init__()
#     def forward(self, in_mat): # in_mat:(400, 96, 5) (batchsize * length_size, pc_num_ti, [x,y,z,velocity,intensity])        
#         return important_index, unimportan_tokens

class PointTransformerLayer(nn.Module):
    def __init__(self, d_model):
        super(PointTransformerLayer, self).__init__()
        self.d_model = d_model
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Generate Q, K, V matrices
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)
        attn = F.softmax(scores, dim=-1)

        # Attend to values
        context = torch.matmul(attn, V)
        out = self.linear_out(context)
        
        return out, attn

class LFPointTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=5):
        super(LFPointTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_linear = nn.Linear(input_dim, d_model)
        self.transformer_layer = PointTransformerLayer(d_model)

    def select_points(self, in_mat, attn):
        batch_size, num_points, _ = in_mat.size()
        selected_points = torch.zeros(batch_size, 96, 5, device=in_mat.device)

        for i in range(batch_size):
            # 取每个点的注意力分数（假设是自注意力的对角线）
            point_attn = attn[i].diagonal(dim1=-2, dim2=-1)  # 取对角线，得到 [96]            
            # 将96个点的注意力分数分成96组，每组4个点
            attn_sums = point_attn.view(-1, 6).sum(dim=1)  # [96]
            # 选出注意力和最高的组的索引
            _, max_group_idx = attn_sums.max(dim=0)
            # 获取对应的点的索引
            group_indices = torch.arange(num_points, device=in_mat.device).view(-1, 6)[max_group_idx]
            # 获取这组点的坐标
            group_points = in_mat[i][group_indices]
            # 计算质心
            centroid = group_points.mean(dim=0)
            # 计算所有点到质心的距离
            all_points = in_mat[i][:, :3]  # 只取前三个维度的数据（点云的坐标）
            distances = torch.norm(all_points - centroid[:3], dim=1)
            # 选择距离最近的16个点
            _, closest_indices = distances.topk(96, largest=False)
            # 保存选择的点
            selected_points[i] = in_mat[i][closest_indices]
        return selected_points

    def forward(self, in_mat):
        # Convert input to d_model size
        x = self.input_linear(in_mat)
        # Pass through the transformer layer
        transformed, attn = self.transformer_layer(x)
        # Select important points
        important_points = self.select_points(in_mat, attn)
        # Return the indices of the most and least important points
        return important_points

class BasePointTiNet(nn.Module):
    def __init__(self):
        super(BasePointTiNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=8, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(8)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(16)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(24)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat): # in_mat:(400, 96, 5) (batchsize * length_size, pc_num_ti, [x,y,z,velocity,intensity])
        x = in_mat.transpose(1,2)   #转置       # x:(400, 5, 96) point(x,y,z,range,intensity,velocity)

        x = self.caf1(self.cb1(self.conv1(x)))  # x:(400, 8, 96)
        x = self.caf2(self.cb2(self.conv2(x)))  # x:(400, 16, 96)
        x = self.caf3(self.cb3(self.conv3(x)))  # x:(400, 24, 96)

        x = x.transpose(1,2)  # x:(400, 96, 24)
        x = torch.cat((in_mat[:,:,:5], x), -1)   # x:(400, 96, 29)  拼接了x,y,z,range
        return x

class GlobalPointTiNet(nn.Module):
    def __init__(self):
        super(GlobalPointTiNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels= 24 + 5,   out_channels=48,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(48)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=48,   out_channels=72,  kernel_size=1)
        self.cb2 = nn.BatchNorm1d(72)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=72, out_channels=96, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(96)
        self.caf3 = nn.ReLU()

        self.attn=nn.Linear(96, 1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        # x:(400, 96, 29)
        x = x.transpose(1,2)   # x:(400, 29, 96)

        x = self.caf1(self.cb1(self.conv1(x)))   # x:(400, 48, 96)
        x = self.caf2(self.cb2(self.conv2(x)))   # x:(400, 72, 96)
        x = self.caf3(self.cb3(self.conv3(x)))   # x:(400, 96, 96)

        x = x.transpose(1,2)   # x:(400, 96, 96)

        attn_weights=self.softmax(self.attn(x))   # attn_weights:(400, 96, 1)
        attn_vec=torch.sum(x*attn_weights, dim=1)  # attn_vec:(400, 96)   * 点乘
        return attn_vec

class GlobalTiRNN(nn.Module):
    def __init__(self):
        super(GlobalTiRNN, self).__init__()
        self.rnn=nn.LSTM(input_size=96, hidden_size=96//2, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True)
        self.fc1 = nn.Linear(96, 16)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x, h0, c0):
        g_vec, (hn, cn)=self.rnn(x, (h0, c0))
        return g_vec

class GlobalTiModule(nn.Module):
    def __init__(self):
        super(GlobalTiModule, self).__init__()
        self.lfpointtransformaernet=LFPointTransformer()
        self.bpointnet=BasePointTiNet()
        self.gpointnet=GlobalPointTiNet()
        self.grnn=GlobalTiRNN()

    def forward(self, x, h0, c0,  batch_size, length_size):
        important_points = self.lfpointtransformaernet(x)       

        x=self.bpointnet(important_points)
        x=self.gpointnet(x)
        x=x.view(batch_size, length_size, 96)
        g_vec=self.grnn(x, h0, c0)
        return g_vec

class ESP_PCT_Net(nn.Module):
    def __init__(self, num_class, hidden_size=96, num_layers=2):
        super(ESP_PCT_Net, self).__init__()
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # parsing
        self.embedding_net = GlobalTiModule()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=96, hidden_size=hidden_size//2, num_layers=num_layers, batch_first=True, bidirectional=True)
        # output layer
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x1, h0, c0, num_class, batch_size,length_size):
        g_vec_parsing0 = self.embedding_net(x1, h0, c0, batch_size, length_size)
        # LSTM layer
        out, _ = self.lstm(g_vec_parsing0)        
        # get the output from the last time step
        out = out[:, -1, :]        
        # output layer
        out = self.fc(out)        
        return F.log_softmax(out, dim=1)

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))
        
if __name__=='__main__':
    batch_size = 16
    print('ESP_PCT_Net:')
    # Instantiate your model with the correct number of classes
    model = ESP_PCT_Net(num_class=36)
    # Generate some dummy input data based on the input size of your model
    data_ti = torch.rand((batch_size * 25, 96, 5), dtype=torch.float32)
    # Initialize the hidden states and cell states
    h0 = torch.zeros((6, batch_size, 96//2), dtype=torch.float32, device='cpu')
    c0 = torch.zeros((6, batch_size, 96//2), dtype=torch.float32, device='cpu')
    # Use the `profile` function from `thop` to calculate FLOPs
    flops, params = profile(model, inputs=(data_ti, h0, c0, 36, batch_size, 25), verbose=False)
    # Format the output to make it more readable
    flops, params = clever_format([flops, params], "%.3f")
    print('FLOPs: {}'.format(flops))
    print('Params: {}'.format(params))
    x=model(data_ti,h0,c0,36,batch_size,25)
    print('\tOutput:', x[0].shape)
    print(model)