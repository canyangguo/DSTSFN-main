# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch import Tensor
import math


class glu(nn.Module):
    def __init__(self, num_of_features):
        super(glu, self).__init__()
        self.num_of_features = num_of_features
        self.weight = nn.Parameter(Tensor(num_of_features, 2 * num_of_features))
        self.bias = nn.Parameter(Tensor(1, 2 * num_of_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight, gain=0.0003)
        torch.nn.init.normal_(self.bias)

    def forward(self, data):
        # (N, B, C) @ (C, 2C) = (N, B, 2C')
        data = torch.matmul(data, self.weight) + self.bias

        # split(N, B, 2C') = (N, B, C'), (N, B, C')
        lhs, rhs = torch.split(data, self.num_of_features, dim=-1)

        # shape is (N, B, C') = (N, B, C') * (N, B, C')
        return lhs * torch.sigmoid(rhs)


class fc(nn.Module):
    def __init__(self, num_of_input, num_of_output):
        super(fc, self).__init__()
        self.weight = nn.Parameter(Tensor(num_of_input, num_of_output))
        self.bias = nn.Parameter(Tensor(1, num_of_output))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight, gain=0.0003)
        torch.nn.init.normal_(self.bias)

    def forward(self, data):
        return torch.matmul(data, self.weight) + self.bias


class mf(nn.Module):
    def __init__(self, merge_step, num_of_vertices, num_of_latents, dropout_rate, adj_st, cand=False):
        super(mf, self).__init__()

        self.num_of_vertices = num_of_vertices
        self.node_emb = nn.Parameter(Tensor(merge_step * num_of_vertices, num_of_latents))
        if cand == False:
            TI = []
            for i in range(merge_step):
                TI.append(torch.eye(num_of_vertices))
            self.TI = torch.stack(TI, dim=1).reshape(num_of_vertices, -1).cuda()
        else:
            self.TI = adj_st

        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.node_emb, gain=0.02)

    def forward(self):
        adj = torch.mm(self.node_emb[-self.num_of_vertices:], self.node_emb.transpose(1, 0)) + self.TI

        adj = self.dropout(self.softmax(adj))

        return adj


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
                 num_of_output=1):
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.predict_length = predict_length

        self.weight1 = nn.Parameter(Tensor(predict_length, input_length * num_of_features, num_of_hidden))
        self.weight2 = nn.Parameter(Tensor(predict_length, num_of_hidden, num_of_output))

        self.pos_emb = position_embedding(predict_length, num_of_vertices, num_of_hidden)

        self.bias1 = nn.Parameter(Tensor(predict_length, 1, num_of_hidden))
        self.bias2 = nn.Parameter(Tensor(predict_length, 1, num_of_output))

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

        data = torch.swapaxes(data, 1, 2)
        data = self.pos_emb(data)

        # (B, N, C') @ (C', T') = (B, N, T'), T' = 1
        data = torch.einsum('btnd, tdc->btnc', data, self.weight2)

        return data


class gcn_operation(nn.Module):
    def __init__(self,
                 num_of_features,
                 num_of_hidden,
                 num_of_vertices):
        super(gcn_operation, self).__init__()

        self.glu_layer = glu(num_of_features)

    def forward(self, wing, data, adj):
        '''
        parameter:
            (3N, B, C): wing shape
            (N, B, C): data shape
            (N, 4N): adj shape
            (N, B, C'): return shape
        '''

        # (3N, B, C) cat (N, B, C) = (4N, B, C)
        data = torch.cat((wing, data), dim=0)

        # (N, 4N) @ (4N, B, C) = (N, B, C)
        data = torch.einsum('nm, mbc->nbc', adj, data)  # gcn_operation

        data = self.glu_layer(data)

        # shape is (N, B, C')
        return data


class stack_gcn(nn.Module):
    def __init__(self, num_of_features, num_of_vertices, num_of_kernel, merge_step):
        super(stack_gcn, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.merge_step = merge_step
        self.gcn_operations = nn.ModuleList([
            gcn_operation(num_of_features,
                          num_of_features,
                          num_of_vertices)
            for i in range(num_of_kernel)])

    def forward(self, data, adj):
        '''
        parameters:
            (N, B, C): data shape
            (3N, B, C): wing shape
            (N, 4N): adj shape
            (N, B, C'): return shape

        operations:
            stack_loop (gcn+glu)
            max pooling
        '''
        need_concat = []

        # (B, 3N, C)
        wing = data[0: (self.merge_step - 1) * self.num_of_vertices]

        # (B, N, C)
        data = data[-self.num_of_vertices:]

        res = data[-self.num_of_vertices:]

        for i, gcn in enumerate(self.gcn_operations):  # multi filters

            # (B, N, C) (gcn + glu) for cat(wing, data):
            data = gcn(wing, data, adj)

            # (1, N, B, C')
            need_concat.append(data)

        # (3, N, B, C'), 3 denote number of gcn filters
        need_concat = torch.stack(need_concat, dim=0)

        # (3, N, B, C'), max pooling for channel wise

        return torch.max(need_concat, dim=0).values + res


class fstgcn(nn.Module):
    def __init__(self,
                 i,
                 current_step,
                 num_of_vertices,
                 num_of_features,
                 merge_step,
                 num_of_kernel,
                 temporal_emb=True,
                 spatial_emb=True):
        super(fstgcn, self).__init__()
        self.current_step = current_step
        self.merge_step = merge_step

        self.conv1 = nn.Conv2d(in_channels=num_of_features,
                               out_channels=num_of_features * 2,
                               kernel_size=(1, merge_step),
                               stride=(1, 1))

        if self.current_step != 12:
            self.conv2 = nn.Conv2d(in_channels=num_of_features,
                                   out_channels=num_of_features * 2,
                                   kernel_size=(1, merge_step + (merge_step - 1) * i),
                                   stride=(1, 1))

        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features

        self.position_embedding = position_embedding(current_step,
                                                     num_of_vertices,
                                                     num_of_features,
                                                     temporal_emb,
                                                     spatial_emb)

        self.stack_gcns = nn.ModuleList([stack_gcn(num_of_features,
                                                   num_of_vertices,
                                                   num_of_kernel,
                                                   merge_step)
                                         for i in range(current_step - (merge_step - 1))])

    def forward(self, original_data, data, adj_st):
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
        data_left, data_right = torch.split(temp, int(temp.shape[1] / 2), dim=1)

        # (B, C, N, T-3) = (B, C, N, T-3) * (B, C, N, T-3)
        data_time_axis = torch.sigmoid(data_left) * data_right

        if self.current_step != 12:
            # (B, C, N, T) <- (B, T, N, C)
            temp = torch.permute(original_data, (0, 3, 2, 1))

            # (B, 2*C, N, T-3) = (B, C, N, T) @ (C, 2C), (1, k), k = 7, 10
            temp = self.conv2(temp)

            # (B, C, N, T-3), (B, C, N, T-3) = split(B, 2 * C, N, T-3)
            data_left, data_right = torch.split(temp, int(temp.shape[1] / 2), dim=1)

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
            t = data[:, i: i + self.merge_step]

            # (B, 4N, C)
            t = torch.reshape(t, (-1, self.merge_step * self.num_of_vertices, self.num_of_features))

            # (4N, B, C) <- (B, 4N, C)
            t = torch.permute(t, (1, 0, 2))

            # (N, B, C') = gcn(glu(4N, B, C)))
            t = layer(t, adj_st)

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


class s2t(nn.Module):
    def __init__(self, num_of_features, input_length, num_of_vertices):
        super(s2t, self).__init__()
        self.position_embedding = position_embedding(input_length,
                                                     num_of_vertices,
                                                     num_of_features)
        self.glu_layer = glu(num_of_features)

    def forward(self, data, temp, adj_s, adj_t):
        res = data
        data = self.position_embedding(data)
        data = torch.einsum('nn, btnc -> btnc', adj_s, data)
        data = torch.einsum('tt, btnc -> btnc', adj_t, data)
        data = self.glu_layer(data)

        # shape is (N, B, C')
        return data + res + temp


class t2s(nn.Module):
    def __init__(self, num_of_features, input_length, num_of_vertices):
        super(t2s, self).__init__()

        self.position_embedding = position_embedding(input_length,
                                                     num_of_vertices,
                                                     num_of_features)
        self.glu_layer = glu(num_of_features)

    def forward(self, data, temp, adj_s, adj_t):
        res = data
        data = self.position_embedding(data)
        data = torch.einsum('tt, btnc -> btnc', adj_t, data)
        data = torch.einsum('ss, btnc -> btnc', adj_s, data)
        data = self.glu_layer(data)

        # shape is (N, B, C')
        return data + res + temp


class _main(nn.Module):
    def __init__(self,
                 adj_st,
                 input_length,
                 num_of_vertices,
                 first_layer_embedding_size,
                 filter_list,
                 use_mask,
                 mask_init_value_st,
                 temporal_emb,
                 spatial_emb,
                 predict_length,
                 num_of_features,
                 receptive_length,
                 ST_adj_dropout_rate,
                 num_of_latents,
                 num_of_gcn_filters):

        super(_main, self).__init__()
        self.adj_st = adj_st
        self.predict_length = predict_length
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.filter_list = filter_list

        self.input_length = input_length

        self.weight = nn.Parameter(Tensor(num_of_features, first_layer_embedding_size))
        self.bias = nn.Parameter(Tensor(1, first_layer_embedding_size))
        self.num_of_features = first_layer_embedding_size
        if use_mask:
            self.mask_st = nn.Parameter(torch.tensor(mask_init_value_st))
        self.stack_times = int((input_length - 1) / (receptive_length - 1))
        self.reduce_length = receptive_length - 1
        self.final_length = input_length - self.stack_times * self.reduce_length

        self.fstgcns = nn.ModuleList([fstgcn(i,
                                             input_length - (receptive_length - 1) * i,
                                             num_of_vertices,
                                             self.num_of_features,
                                             receptive_length,
                                             num_of_gcn_filters,
                                             temporal_emb=temporal_emb,
                                             spatial_emb=spatial_emb)
                                      for i in range(self.stack_times)])

        self.output_layer = output_layer(num_of_vertices,
                                         self.final_length,
                                         self.num_of_features,
                                         predict_length)

        #self.mf = mf(receptive_length, num_of_vertices, num_of_latents, ST_adj_dropout_rate, adj_st)
        S_adj_dropout_rate = 1 - (1 - ST_adj_dropout_rate) * receptive_length
        self.mf_s2 = mf(1, num_of_vertices, num_of_latents, S_adj_dropout_rate, adj_st[:, -num_of_vertices:])
        self.mf_s3 = mf(1, num_of_vertices, num_of_latents, S_adj_dropout_rate, adj_st[:, -num_of_vertices:])

        self.mf_t2 = nn.Parameter(Tensor(input_length, input_length))
        self.mf_t3 = nn.Parameter(Tensor(input_length, input_length))
        self.I = torch.eye(input_length).cuda()

        self.s2t = nn.ModuleList([
            s2t(self.num_of_features, input_length, num_of_vertices)
            for i in range(self.stack_times)
        ])

        self.t2s = nn.ModuleList([
            t2s(self.num_of_features, input_length, num_of_vertices)
            for i in range(self.stack_times)
        ])

        self.conv2 = nn.Conv2d(in_channels=input_length,
                               out_channels=3,
                               kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=input_length,
                               out_channels=3,
                               kernel_size=(1, 1))



        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.mf_t2, gain=0.0003)
        torch.nn.init.xavier_normal_(self.mf_t3, gain=0.0003)
        torch.nn.init.xavier_normal_(self.weight, gain=0.0003)
        torch.nn.init.normal_(self.bias)




    def forward(self, data, x_day_index, x_time_index, y_day_index, y_time_index):

        '''
        (B, T, N, F): data shape
        (B, T, N): return shape
        (N, 4N): adj shape
        '''

        # (B, T, N, C) = (B, T, N, F) * (F, C)
        data = torch.matmul(data, self.weight) + self.bias  # first layer embedding
        
        adj = self.mask_st * self.adj_st  # mask adj
        # (N, 4N) = (N, 4N) * (N, 4N)
        adj_s2 = self.mf_s2()
        adj_t2 = self.mf_t2 + self.I
        adj_s3 = self.mf_s3()
        adj_t3 = self.mf_t3 + self.I
        #adj = self.mf()

        # (B, T, N, C), T = 12
        temp, x1, x2, x3 = data, data, data, data

        for i in range(self.stack_times):
            x1 = self.fstgcns[i](temp, x1, adj)
            x2 = self.s2t[i](x2, temp, adj_s2, adj_t2)
            x3 = self.t2s[i](x3, temp, adj_s3, adj_t3)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        '''
        for i in range(self.stack_times):
            x1 = self.fstgcns[i](temp, x1, adj)

        for _, s2t in enumerate(self.s2t):
            # (B, T, N, C)
            x2 = s2t(x2, temp, adj_s2, adj_t2)
        x2 = self.conv2(x2)

        for _, t2s in enumerate(self.t2s):
            # (B, T, N, C)
            x3 = t2s(x3, temp, adj_s3, adj_t3)
        x3 = self.conv3(x3)
        '''
        data = x1 + x2 + x3
        data = self.output_layer(data)

        # (B, T, N)
        data = data.squeeze(dim=3)
        return data




