import torch
import torch.nn as nn
import math


class weight_and_initial(nn.Module):
    def __init__(self, input_dim, output_dim, num=1, bias=False):
        super(weight_and_initial, self).__init__()

        if num == 1:
            self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
            if bias == True:
                self.bias = nn.Parameter(torch.empty(1, output_dim))
            else:
                self.bias = None

        else:
            self.weight = nn.Parameter(torch.empty(num, input_dim, output_dim))
            if bias == True:
                self.bias = nn.Parameter(torch.empty(num, 1, output_dim))
            else:
                self.bias = None

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias == True:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self):
        if self.bias is not None:
            return self.weight, self.bias
        return self.weight


class glu(nn.Module):
    def __init__(self, d_model, receptive_length=1, type="linear"):
        super(glu, self).__init__()
        self.d_model = d_model
        self.type = type
        if type == "linear":
            self.fc = nn.Linear(d_model, 2 * d_model)
        if type == "conv":
            self.conv = nn.Conv2d(d_model, 2 * d_model, (1, receptive_length), stride=(1, 1))

    def forward(self, x):
        # (B, T, N, C) @ (C, 2C) = B, T, N, 2C
        if self.type == "linear":
            x = self.fc(x)
        if self.type == "conv":
            x = x.permute(0, 3, 2, 1)
            x = self.conv(x)
            x = x.permute(0, 3, 2, 1)

        # split(B, T, N, 2C) = (B, T, N, C), (B, T, N, C)
        lhs, rhs = torch.split(x, self.d_model, dim=-1)
        return lhs * torch.sigmoid(rhs)


class feed_forward_network(nn.Module):
    def __init__(self, d_model):
        super(feed_forward_network, self).__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_model)
        self.fc2 = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class matrix_decomposition(nn.Module):
    def __init__(self, d1, d2, r, num=1):
        super(matrix_decomposition, self).__init__()

        self.emb1 = weight_and_initial(d1, r, num)

        self.emb2 = weight_and_initial(d2, r)

    def forward(self):
        return torch.matmul(self.emb1(), self.emb2().transpose(0, 1))  # num, d1, d2



class tucker_decomposition(nn.Module):
    def __init__(self, d1, d2, d3, r1, r2, r3, re=False):
        super(tucker_decomposition, self).__init__()
        self.re = re
        self.d2 = d2
        self.d3 = d3
        self.emb1 = nn.Parameter(torch.empty(d1, r1))
        self.emb2 = nn.Parameter(torch.empty(d2, r2))
        self.emb3 = nn.Parameter(torch.empty(d3, r3))
        self.core = nn.Parameter(torch.empty(r1, r2, r3))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.emb1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.emb2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.emb3, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.core, a=math.sqrt(5))

    def forward(self):
        # if self.d3 % self.d2 != 0:
        #     emb2 = self.emb2
        # else:
        #     emb2 = self.emb3[-self.d2:]
        x = torch.einsum('ax, xyz -> ayz', self.emb1, self.core)
        x = torch.einsum('by, ayz -> abz', self.emb2, x)
        x = torch.einsum('cz, abz -> abc', self.emb3, x)
        if self.re == False:
            return x
        else:
            return x, self.emb1



class position_embedding(nn.Module):
    def __init__(self, num_nodes, num_times, num_days, input_len, d_model, node_pos, time_pos, day_pos, step_pos):
        super(position_embedding, self).__init__()
        self.node_emb = None
        self.time_emb = None
        self.day_emb = None
        self.step_emb = None
        if node_pos:
            self.node_emb = weight_and_initial(num_nodes, d_model)
        if time_pos:
            self.time_emb = weight_and_initial(num_times, d_model)
        if day_pos:
            self.day_emb = weight_and_initial(num_days, d_model)
        if step_pos:
            self.step_emb = weight_and_initial(input_len, d_model)

    def forward(self, x, ti, di):
        if self.node_emb is not None:
            x += self.node_emb().unsqueeze(0).unsqueeze(0)
        if self.time_emb is not None:
            x += self.time_emb()[ti].unsqueeze(1).unsqueeze(1)
        if self.day_emb is not None:
            x += self.day_emb()[di].unsqueeze(1).unsqueeze(1)
        if self.step_emb is not None:
            x += self.step_emb().unsqueeze(0).unsqueeze(2)
        return x






