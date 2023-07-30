import torch
import torch.nn as nn
import torch.nn.functional as F

class Diff(nn.Module):
    def __init__(self):
        super(Diff, self).__init__()

    def forward(self, x):
        return torch.unsqueeze(x[:, 1] - x[:, 0], dim=-1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class AttentionMask1(nn.Module):
    def __init__(self, a):
        super(AttentionMask1, self).__init__()
        self.relu = nn.ReLU()
        # self.activ = a()
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        r_out = self.relu(x)
        # a_out = self.activ(r_out)
        att_mask = self.soft(r_out)
        elemwise_prod = att_mask * r_out
        
        return elemwise_prod

class GWANNet5(nn.Module):
    """Attention applied to GroupTrain model after the encoding created
    by the CNN layer and Avg Pooling.

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self, grp_size, enc, inp, h, d, out, activation, 
                 att_model, att_activ):
        super(GWANNet5, self).__init__()

        self.snp_enc = nn.Conv1d(inp, enc, 1)
        self.pool_ind = nn.AvgPool1d(grp_size)        
        self.att_mask = att_model(att_activ)
        # self.pool_features = nn.AdaptiveMaxPool1d(32)
        self.snps_model = nn.Sequential(
            BasicNN(enc, h, d, out, activation),
            nn.ReLU(),
            nn.BatchNorm1d(out))
        self.att_out = None
        self.num_snps = inp
        
        # For T2D and AD
        self.end_model = BasicNN(16, [16, 8], [0.1, 0.1], 2, nn.ReLU)
        
    def forward(self, x):
        snps_enc = self.snp_enc(torch.transpose(x[:, :, :self.num_snps], 1, 2))
        snps_pooled = torch.squeeze(self.pool_ind(snps_enc), dim=-1)
        self.att_out = self.att_mask(snps_pooled)
        # self.att_out.requires_grad_(True)
        # features_pooled = self.pool_features(
        #     torch.unsqueeze(self.att_out, dim=1))
        snps_out = self.snps_model(torch.squeeze(self.att_out, dim=1))
        
        cov_out = torch.mean(x[:, :, self.num_snps:], dim=1)
        
        data_vec = torch.cat((snps_out, cov_out), dim=-1)
        
        raw_out = self.end_model(data_vec)
        
        return raw_out

class GroupAttention(nn.Module):
    """Attention applied to GroupTrain model after the encoding created
    by the CNN layer and Avg Pooling.

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self, grp_size, inp, enc, h, d, out, activation, 
            att_model, att_activ):
        super(GroupAttention, self).__init__()

        self.grp_enc = nn.Conv1d(inp, enc, 1)
        self.pool = nn.AvgPool1d(grp_size)
        self.att_mask = att_model(att_activ)
        self.end_model = BasicNN(enc, h, d, out, activation)
        self.att_out = None
        
    def forward(self, x):
        eout = self.grp_enc(torch.transpose(x, 1, 2))
        # self.att_out = self.att_mask(eout)
        # self.att_out.requires_grad_(True)
        # pooled = torch.squeeze(self.pool(self.att_out), dim=-1)
        # raw_out = self.end_model(pooled)
        
        pooled = torch.squeeze(self.pool(eout), dim=-1)
        self.att_out = self.att_mask(pooled)
        # self.att_out.requires_grad_(True)
        raw_out = self.end_model(self.att_out)
        
        return raw_out

class BasicNN(nn.Module):

    def __init__(self, inp, h, d, out, activation):
        super(BasicNN, self).__init__()

        assert(len(h) == len(d))
        
        inp = [nn.Linear(inp, h[0])]
        op = [nn.Linear(h[-1], out)]
        hidden = [nn.Linear(h[i-1], h[i]) for i in range(1, len(h))]
        self.linears = nn.ModuleList(inp + hidden + op)
        self.dropouts = nn.ModuleList([nn.Dropout(prob) for prob in d])
        self.bnorms = nn.ModuleList([nn.BatchNorm1d(inp) for inp in h])
        self.activation = nn.ModuleList([activation() for i in range(len(h))])
        
    def forward(self, x):
        X = self.linears[0](x)
        for l, drop, bnorm, activ in zip(self.linears[1:], self.dropouts, 
                                         self.bnorms, self.activation):
            
            # X = l(bnorm(activ(drop(X))))
            X = l(drop(bnorm(activ(X))))
            
        return X
