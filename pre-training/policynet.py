
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, batchsize, bias=False):

        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batchsize = batchsize
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def forward(self, input, adj):

        support = torch.einsum("jik,kp->jip",input,self.weight)
        if self.bias is not None:
            support = support + self.bias
        support = torch.reshape(support,[support.size(0),-1])
        output = torch.spmm(adj, support)
        output = torch.reshape(output,[output.size(0),self.batchsize,-1])
        return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class PolicyNet(nn.Module):
    
    def __init__(self,args,statedim):

        super(PolicyNet, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList()
        for i in range(len(args.pnhid)):
            if (i == 0):
                self.gcn.append(GraphConvolution(statedim, args.pnhid[i], args.batchsize, bias=True))
            else:
                self.gcn.append(GraphConvolution(args.pnhid[i - 1], args.pnhid[i], args.batchsize, bias=True))

        self.output_layer = nn.Linear(args.pnhid[-1], 1, bias=False)


    def forward(self, state, adj):

        x = state.transpose(0, 1)
        for layer in self.gcn:
            x = F.relu(layer(x, adj))
        x = self.output_layer(x).squeeze(-1).transpose(0, 1)
        return x


class PolicyNet2(nn.Module):

    def __init__(self, args, statedim):
        super(PolicyNet2,self).__init__()
        self.args = args
        self.lin1 = nn.Linear(statedim,args.pnhid[0])
        self.lin2 = nn.Linear(args.pnhid[0],args.pnhid[0])
        self.lin3 = nn.Linear(args.pnhid[0], 1)
        stdv = 1. / math.sqrt(self.lin1.weight.size(1))

        self.lin1.weight.data.uniform_(-stdv, stdv)
        self.lin2.weight.data.uniform_(-stdv, stdv)
        self.lin3.weight.data.uniform_(-stdv, stdv)

        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.zeros_(self.lin3.bias)


    def forward(self,state,adj):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        logits = self.lin3(x).squeeze(-1)
        return logits
