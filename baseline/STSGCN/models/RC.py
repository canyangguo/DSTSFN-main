# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch import Tensor
import math


class multi_head_attention(nn.Module):
    def __init__(self, num_of_vertices, num_of_features, num_of_hidden, k_head):
        super(multi_head_attention, self).__init__()
        self.wq = nn.Parameter(Tensor(num_of_features, k_head*num_of_hidden))
        self.wk = nn.Parameter(Tensor(num_of_features, k_head*num_of_hidden))
        self.wv = nn.Parameter(Tensor(num_of_features, k_head*num_of_hidden))
        self.num_of_hidden = num_of_hidden
        self.num_of_vertices = num_of_vertices
        self.k_head = k_head
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(num_of_hidden)
        self.weight = nn.Parameter(Tensor(k_head*num_of_hidden, num_of_features))
        self.bias = nn.Parameter(Tensor(1, num_of_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.wq, gain=0.0003)
        torch.nn.init.xavier_normal_(self.wk, gain=0.0003)
        torch.nn.init.xavier_normal_(self.wv, gain=0.0003)
        torch.nn.init.xavier_normal_(self.weight, gain=0.0003)
        torch.nn.init.normal_(self.bias)


    def forward(self, x):

        res = x[3*self.num_of_vertices: 4*self.num_of_vertices]
        # 4*N, B, K, C'
        Q = torch.einsum('mbc, cd -> mbd', x, self.wq)\
            .reshape(x.shape[0], -1, self.k_head, self.num_of_hidden)

        # N, B, K, C'
        K = torch.einsum('nbc, cd -> nbd', x[3*self.num_of_vertices: 4*self.num_of_vertices], self.wk)\
            .reshape(self.num_of_vertices, -1, self.k_head, self.num_of_hidden)

        # 4*N, B, K, C'
        V = torch.einsum('mbc, cd -> mbd', x, self.wv)\
            .reshape(x.shape[0], -1, self.k_head, self.num_of_hidden)

        score = torch.einsum('mbkd, nbkd -> mnbk', Q, K) / (self.num_of_hidden**0.5)

        attn = self.softmax(score)

        # N, B, K, D
        x = torch.einsum('mbkd, mnbk -> nbkd', V, attn)
        x = x.reshape(self.num_of_vertices, -1, self.k_head*self.num_of_hidden)

        x = torch.einsum('nbd, dc -> nbc', x, self.weight) + self.bias

        return self.layer_norm(x + res)



class position_embedding(nn.Module):
    def __init__(self,
                 input_length,
                 num_of_vertices,
                 embedding_size,
                 temporal=True,
                 spatial=True,
                 embedding_type='add'):
        super(position_embedding, self).__init__()
        self.embedding_type = embedding_type

        if temporal:
            # shape is (1, T, 1, C)
            self.temporal_emb = nn.Parameter(Tensor(1, input_length, 1, embedding_size))

        if spatial:
            # shape is (1, 1, N, C)
            self.spatial_emb = nn.Parameter(Tensor(1, 1, num_of_vertices, embedding_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.temporal_emb, gain=0.0003)
        torch.nn.init.xavier_normal_(self.spatial_emb, gain=0.0003)


    def forward(self, data):
        '''
        parameter:
            (B, T, N, C): input shape
            (B, T, N, C): return shape
            (1, T, 1, C): temporal embedding shape
            (1, 1, N, C): spatial embedding shape
            'add' or 'multiply': embedding_type
        '''

        # (B, T, N, C)
        if self.embedding_type == 'add':
            if self.temporal_emb is not None:

                # (B, T, N, C) = (B, T, N, C) + (1, T, 1, C)
                data = data + self.temporal_emb

            if self.spatial_emb is not None:

                # (B, T, N, C) = (B, T, N, C) + (1, 1, N, C)
                data = data + self.spatial_emb

        elif self.embedding_type == 'multiply':
            if self.temporal_emb is not None:

                # (B, T, N, C) = (B, T, N, C) * (1, T, 1, C)
                data = data * self.temporal_emb
            if self.spatial_emb is not None:

                # (B, T, N, C) = (B, T, N, C) * (1, 1, N, C)
                data = data * self.spatial_emb

        return data


class output_layer(nn.Module):
    def __init__(self,
                 num_of_vertices,
                 input_length,  # T = 3
                 num_of_features,
                 predict_length,
                 num_of_hidden=128,
                 T = 12):

        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.T = T

        self.weight1 = nn.Parameter(Tensor(T, input_length * num_of_features, num_of_hidden))
        self.weight2 = nn.Parameter(Tensor(T, num_of_hidden, predict_length))

        self.bias1 = nn.Parameter(Tensor(T, 1, num_of_hidden))
        self.bias2 = nn.Parameter(Tensor(T, 1, predict_length))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight1, gain=0.0003)
        torch.nn.init.xavier_normal_(self.weight2, gain=0.0003)

        torch.nn.init.normal_(self.bias1)
        torch.nn.init.normal_(self.bias2)

    def forward(self, data):
        # data shape is (B, N, T, C)
        data = torch.swapaxes(data, 1, 2)

        # (B, N, T * C), T = 3
        data = torch.reshape(data, (-1, self.num_of_vertices, self.input_length * self.num_of_features))

        # (B, N, T * C) @ (T * C, C') = (B, N, C')
        data = torch.relu(torch.einsum('bnc, tcd->bntd', data, self.weight1))

        # (B, N, C') @ (C', T') = (B, N, T'), T' = 1
        data = torch.einsum('bntd, tdc->bntc', data, self.weight2)

        data = data.reshape(-1, self.num_of_vertices, self.T)

        data = torch.swapaxes(data, 1, 2)
        # (B * T', N) <- (B, N, T')
        return data


class gcn_operation(nn.Module):
    def __init__(self,
                 k_head,
                 num_of_features,
                 num_of_hidden,
                 num_of_vertices,
                 activation='GLU'):
        super(gcn_operation, self).__init__()
        self.mha = multi_head_attention(num_of_vertices, num_of_features, num_of_hidden, k_head)

        self.activation = activation

        self.weight = nn.Parameter(Tensor(k_head*num_of_features, 2 * num_of_hidden))
        self.bias = nn.Parameter(Tensor(1, 2*num_of_hidden))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight, gain=0.0003)
        torch.nn.init.normal_(self.bias)


    def forward(self, data):
        '''
        parameter:
            (3N, B, C): wing shape
            (N, B, C): data shape
            (N, 4N): adj shape
            (N, B, C'): return shape
        '''

        assert self.activation in {'GLU', 'relu'}

        # N, B, K*D
        data = self.mha(data)

        # # (N, B, C) @ (C, 2C) = (N, B, 2C')
        # data = torch.matmul(data, self.weight) + self.bias
        #
        # # split(N, B, 2C') = (N, B, C'), (N, B, C')
        # lhs, rhs = torch.split(data, int(data.shape[2] / 2), dim=2)
        #
        # # shape is (N, B, C') = (N, B, C') * (N, B, C')
        # data = lhs * torch.sigmoid(rhs)

        # shape is (N, B, C')
        return data


class stack_gcn(nn.Module):
    def __init__(self, filters, num_of_features, num_of_vertices, activation):
        super(stack_gcn, self).__init__()
        self.num_of_vertices = num_of_vertices

        self.gcn_operation = gcn_operation(len(filters),
                             num_of_features,
                             num_of_features,
                             num_of_vertices,
                             activation=activation)

    def forward(self, data):

        data = self.gcn_operation(data)
        return data


class fstgcn(nn.Module):
    def __init__(self,
                 T,
                 num_of_vertices,
                 num_of_features,
                 filters,
                 res_filters,
                 activation,
                 temporal_emb=True,
                 spatial_emb=True):
        super(fstgcn, self).__init__()
        self.T = T

        self.conv1 = nn.Conv2d(in_channels=num_of_features,
                               out_channels=num_of_features*2,
                               kernel_size=(1, res_filters[0]),
                               stride=(1, 1))

        if self.T != 12:
            self.conv2 = nn.Conv2d(in_channels=num_of_features,
                                   out_channels=num_of_features*2,
                                   kernel_size=(1, res_filters[1]),
                                   stride=(1, 1))

        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features

        self.position_embedding = position_embedding(T,
                                                     num_of_vertices,
                                                     num_of_features,
                                                     temporal_emb,
                                                     spatial_emb)

        self.stack_gcns = nn.ModuleList([stack_gcn(filters,
                                                             num_of_features,
                                                             num_of_vertices,
                                                             activation=activation)
                                                             for i in range(T - 3)])

    def forward(self, original_data, data):
        '''
        parameter:
            (B, T, N, C): original data shape, T = 12
            (B, T, N, C): data shape, T = 12, 9, 6
            (4N, N): adj_st shape
            (B, T-3, N, C): return shape, T = 9, 6, 3
        operation:
            multi conv_res
            positioning_embedding
            loop(gcn + glu)
        '''

        '''----------res_conv begin----------'''
        # (B, C, N, T) <- (B, T, N, C)
        temp = torch.permute(data, (0, 3, 2, 1))

        # (B, 2*C, N, T-3) = (B, C, N, T) @ (C, 2C), (1, k), k = 4
        temp = self.conv1(temp)

        # (B, C, N, T-3), (B, C, N, T-3) = split(B, 2 * C, N, T-3)
        data_left, data_right = torch.split(temp, int(temp.shape[1]/2), dim=1)

        # (B, C, N, T-3) = (B, C, N, T-3) * (B, C, N, T-3)
        data_time_axis = torch.sigmoid(data_left) * data_right

        if self.T != 12:
            # (B, C, N, T) <- (B, T, N, C)
            temp = torch.permute(original_data, (0, 3, 2, 1))

            # (B, 2*C, N, T-3) = (B, C, N, T) @ (C, 2C), (1, k), k = 7, 10
            temp = self.conv2(temp)

            # (B, C, N, T-3), (B, C, N, T-3) = split(B, 2 * C, N, T-3)
            data_left, data_right = torch.split(temp, int(temp.shape[1]/2), dim=1)

            # (B, C, N, T-3) += (B, C, N, T-3) * (B, C, N, T-3)
            data_time_axis += torch.sigmoid(data_left) * data_right

        # (B, T-3, N, C) <- (B, C, N, T-3)
        data_res = torch.permute(data_time_axis, (0, 3, 2, 1))
        '''----------res_conv end----------'''

        # (B, T, N, C)
        data = self.position_embedding(data)

        need_concat = []
        for i, layer in enumerate(self.stack_gcns):

            # (B, 4, N, C)
            t = data[:, i: i + 4]

            # (B, 4N, C)
            t = torch.reshape(t, (-1, 4 * self.num_of_vertices, self.num_of_features))

            # (4N, B, C) <- (B, 4N, C)
            t = torch.permute(t, (1, 0, 2))

            # (N, B, C') = gcn(glu(4N, B, C)))
            t = layer(t)

            # (B, N, C')
            t = torch.swapaxes(t, 0, 1)

            # (1, B, N, C')
            need_concat.append(t)

        # (B, T - 3, N, C')
        need_concat = torch.stack(need_concat, dim=1)

        # (B, T-3, N, C') = (B, T-3, N, C') + (B, T-3, N, C')
        data = need_concat + data_res

        # (B, T-3, N, C')
        return data


class _main(nn.Module):
    def __init__(self,
                 adj_st,
                 input_length,
                 num_of_vertices,
                 first_layer_embedding_size,
                 filter_list,
                 res_filters,
                 activation,
                 use_mask,
                 mask_init_value_st,
                 temporal_emb,
                 spatial_emb,
                 num_of_features,
                 predict_length=12,
                 L1loss=0,
                 k=4):

        super(_main, self).__init__()
        self.adj_st = adj_st
        self.predict_length = predict_length
        self.num_of_features = num_of_features
        self.filter_list = filter_list
        self.L1loss = L1loss

        self.weight = nn.Parameter(Tensor(num_of_features, first_layer_embedding_size))
        self.bias = nn.Parameter(Tensor(1, first_layer_embedding_size))

        self.num_of_features = first_layer_embedding_size

        self.fstgcns = nn.ModuleList([fstgcn(input_length-3*i,
                                             num_of_vertices,
                                             self.num_of_features,
                                             filters,
                                             res_filters[i],
                                             activation=activation,
                                             temporal_emb=temporal_emb,
                                             spatial_emb=spatial_emb)
                                      for i, filters in enumerate(filter_list)])



        self.output_layer = output_layer(num_of_vertices,
                                         input_length-3*len(filter_list),
                                         self.num_of_features,
                                         predict_length=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight, gain=0.0003)
        torch.nn.init.normal_(self.bias)

    def forward(self, data):

        '''
        (B, T, N, F): data shape
        (B, T, N): return shape
        (N, 4N): adj shape
        '''

        # (B, T, N, C) = (B, T, N, F) * (F, C)
        data = torch.matmul(data, self.weight) + self.bias # first layer embedding


        # (B, T, N, C), T = 12
        temp = data  # store original data

        for _, fstgcn in enumerate(self.fstgcns):
            # (B, T-3, N, C)
            data = fstgcn(temp, data)


        data = self.output_layer(data)
        # (B, T, N)
        return data, self.L1loss



