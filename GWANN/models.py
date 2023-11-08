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
    def __init__(self):
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

class GWANNet5(torch.nn.Module):
    """
    """
    def __init__(self, grp_size, enc, snps, cov_model, h, d, out, activation, 
                 att_model):
        super(GWANNet5, self).__init__()

        self.grp_size = grp_size
        self.snp_enc = nn.Conv1d(snps, enc, 1)
        self.snp_pool = nn.AvgPool1d(grp_size)        
        self.snp_mask = att_model()
        self.snp_model = nn.Sequential(
            BasicNN(enc, h, d, out, activation),
            nn.ReLU(),
            nn.BatchNorm1d(out))
        
        self.cov_model = cov_model

        self.num_snps = snps
        
        # For T2D and AD
        self.end_model = BasicNN(16, [16, 8], [0.1, 0.1], 1, nn.ReLU)

    def forward(self, x):
        snp_enc = self.snp_enc(torch.transpose(x[:, :, :self.num_snps], 1, 2))
        snp_pooled = torch.squeeze(self.snp_pool(snp_enc), dim=-1)
        snp_att_out = self.snp_mask(snp_pooled)
        snp_out = self.snp_model(torch.squeeze(snp_att_out, dim=1))
        
        cov_out = self.cov_model(x[:, :, self.num_snps:])

        data_vec = torch.cat((snp_out, cov_out), dim=-1)
        
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
            att_model):
        super(GroupAttention, self).__init__()

        self.grp_size = grp_size
        self.grp_enc = nn.Conv1d(inp, enc, 1)
        self.pool = nn.AvgPool1d(grp_size)
        self.att_mask = att_model()
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

class MLP(nn.Module):
    class Block(nn.Module):
        def __init__(self, d_in:int, d_out:int, activation:str, dropout:float):
            super().__init__()
            self.linear = nn.Linear(d_in, d_out)
            self.activation = getattr(nn, activation)()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout(self.activation(self.linear(x)))
            
    def __init__(self, d_in:int, d_out:int, d_layers:list, dropout:float=0.1, 
                    activation:str='ReLU'):
        super(MLP, self).__init__()

        self.blocks = nn.Sequential(
            *[
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    activation=activation,
                    dropout=dropout,
                )
                for i, d in enumerate(d_layers)
            ]
        )
        
        self.head = None
        if d_out >= 1:
            self.head = nn.Linear(d_layers[-1], d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x

class BranchedMLP(torch.nn.Module):
    def __init__(self, d_snps:int, d_layers:list, d_out:int, 
                 cov_branch:MLP, dropout:float=0.1, 
                 activation:str='ReLU'):
        super(BranchedMLP, self).__init__()

        self.d_snps = d_snps
        self.snp_branch = MLP(d_snps, -1, d_layers, dropout, activation)
        self.cov_branch = cov_branch
        self.cov_branch.head = None
        cov_model_out = self.cov_branch.blocks[-1].linear.out_features

        self.head = nn.Linear(d_layers[-1] + cov_model_out, d_out)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        snp_enc = self.snp_branch(x[:, :self.d_snps])
        cov_enc = self.cov_branch(x[:, self.d_snps:])
        out = self.head(torch.cat((snp_enc, cov_enc), dim=-1))
        return out
