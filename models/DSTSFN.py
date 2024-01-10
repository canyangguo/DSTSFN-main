import torch
import torch.nn as nn
import math
from models.basic_module import weight_and_initial, glu, matrix_decomposition


class dynamic_temporal_graph_convolution(nn.Module):
    def __init__(self, d_model, in_num_of_vertices, out_num_of_vertices, input_length, output_length, num_of_latents, num_of_times, num_of_days, drop_rate=0.0):
        super(dynamic_temporal_graph_convolution, self).__init__()
        self.sta_adj = weight_and_initial(input_length, output_length)
        self.t_adjs = weight_and_initial(input_length, output_length, num_of_times)
        self.d_adjs = weight_and_initial(input_length, output_length, num_of_days)
        self.glu = glu(d_model)

    def forward(self, x, ti, di, ep, tdx, S2D):

        sta_adj = self.sta_adj()
        if ep < S2D:
            x = torch.einsum('pq, bqnc -> bpnc', sta_adj[: tdx], x)
        else:
            t_adj = self.t_adjs()[ti]
            d_adj = self.d_adjs()[di]
            adj = t_adj/3 + d_adj/3 + sta_adj/3

            x = torch.einsum('bpq, bqnc -> bpnc', adj[:, : tdx], x)

        x = self.glu(x)
        return x

class dynamic_spatial_graph_convolution(nn.Module):
    def __init__(self, d_model, in_num_of_vertices, out_num_of_vertices, input_length, output_length, num_of_latents, num_of_times, num_of_days, drop_rate=0.8):
        super(dynamic_spatial_graph_convolution, self).__init__()
        self.sta_adj = matrix_decomposition(in_num_of_vertices, out_num_of_vertices, num_of_latents)
        self.t_adjs = matrix_decomposition(in_num_of_vertices, out_num_of_vertices, num_of_latents, num_of_times)
        self.d_adjs = matrix_decomposition(in_num_of_vertices, out_num_of_vertices, num_of_latents, num_of_days)

        self.drop = nn.Dropout(drop_rate)
        self.glu = glu(d_model)

    def forward(self, x, ti, di, ep, S2D):
        sta_adj = self.sta_adj()

        if ep < S2D:
            x = torch.einsum('mn, btnc -> btmc', self.drop(sta_adj), x)
        else:
            t_adj = self.t_adjs()[ti]
            d_adj = self.d_adjs()[di]

            adj = t_adj/3 + d_adj/3 + sta_adj/3
            adj = self.drop(adj)

            x = torch.einsum('bmn, btnc -> btmc', adj, x)  # 是否在这加res

        x = self.glu(x)
        return x


class encoder(nn.Module):
    def __init__(self, d_model, in_num_of_vertices, out_num_of_vertices, input_length, output_length, num_of_times, num_of_days, num_of_layers, num_of_latents):
        super(encoder, self).__init__()
        self.st = nn.ModuleList([
            st_module(d_model, in_num_of_vertices, out_num_of_vertices, input_length, output_length, num_of_latents, num_of_times, num_of_days)
            for i in range(num_of_layers)
        ])
    def forward(self, x, ti, di, ep, tdx, S2D):
        for st in self.st:
            x = st(x, ti, di, ep, tdx, S2D)
        return x


class st_module(nn.Module):
    def __init__(self, d_model, in_num_of_vertices, out_num_of_vertices, input_length, output_length, num_of_latents, num_of_times, num_of_days):
        super(st_module, self).__init__()

        self.psgcn = dynamic_spatial_graph_convolution(d_model, in_num_of_vertices, out_num_of_vertices, input_length, output_length, num_of_latents, num_of_times, num_of_days)
        self.ptgcn = dynamic_temporal_graph_convolution(d_model, in_num_of_vertices, out_num_of_vertices, input_length, output_length, num_of_latents, num_of_times, num_of_days)


    def forward(self, x, ti, di, ep, tdx, S2D):

        x = self.psgcn(x, ti, di, ep, S2D) + x

        x = torch.cat((self.ptgcn(x, ti, di, ep, tdx, S2D) + x[:, :tdx], x[:, tdx:]), dim=1)

        return x


class make_model(nn.Module):
    def __init__(self, input_length, num_of_vertices, d_model, filter_list, use_mask,
                 temporal_emb, spatial_emb, predict_length, num_of_features, num_of_outputs, receptive_length,
                 dropout_rate, num_of_latents, num_of_layers, num_of_times=288, num_of_days=7, drop_rate=0.8):
        super(make_model, self).__init__()
        self.data_emb = nn.Linear(num_of_features, d_model)
        self.encoder = encoder(d_model, num_of_vertices, num_of_vertices, input_length, predict_length, num_of_times, num_of_days, num_of_layers, num_of_latents)
        self.reg = nn.Linear(d_model, num_of_outputs)

    def forward(self, x, ti, di, ep=200, tdx=12, S2D=0):


        x = self.data_emb(x)

        x = self.encoder(x, ti, di, ep, tdx, S2D)[:, : tdx]

        x = self.reg(x).squeeze(-1)
        return x