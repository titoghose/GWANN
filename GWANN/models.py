import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class AttentionMask1(nn.Module):
    def __init__(self, a):
        super(AttentionMask1, self).__init__()
        self.relu = nn.ReLU()
        self.activ = a()
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        r_out = self.relu(x)
        a_out = self.activ(r_out)
        att_mask = self.soft(a_out)
        elemwise_prod = att_mask * r_out
        
        return elemwise_prod

class AttentionMask2(nn.Module):
    def __init__(self, a):
        super(AttentionMask2, self).__init__()
        self.relu = nn.ReLU()
        self.activ = a()
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        a_out = self.activ(x)
        att_mask = self.soft(a_out)
        elemwise_prod = att_mask * x
        r_out = self.relu(elemwise_prod)
        
        return r_out

class AttentionMask3(nn.Module):
    def __init__(self, a):
        super(AttentionMask3, self).__init__()
        self.activ = a()
        self.soft = nn.Softmax(dim=-2)
        
    def forward(self, x):
        a_out = self.activ(x)
        att_mask = self.soft(a_out)
        elemwise_prod = att_mask * x
        
        return elemwise_prod

class AttentionMask4(nn.Module):
    def __init__(self, a):
        super(AttentionMask4, self).__init__()
        self.activ = a()
        self.soft = nn.Softmax(dim=-2)
        
    def forward(self, x):
        a_out = self.activ(x)
        att_mask = self.soft(a_out)
        elemwise_prod = att_mask * x
        
        return elemwise_prod

class GWANNet(nn.Module):
    """Attention applied to GroupTrain model after the encoding created
    by the CNN layer and Avg Pooling.

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self, grp_size, enc, h, d, out, activation, 
            num_snps, att_model, att_activ, cov_model):
        super(GWANNet, self).__init__()

        self.cov_model = cov_model
        self.snp_enc = nn.Conv1d(num_snps, enc, 1)
        self.pool_ind = nn.AvgPool1d(grp_size)
        self.pool_features = nn.AdaptiveAvgPool1d(32)
        self.att_mask = att_model(att_activ)
        self.snps_model = BasicNN(32, h, d, out, activation)
        self.att_out = None
        self.num_snps = num_snps    

        self.end_model = BasicNN(32, [16, 8], [0.1, 0.1], 2, nn.ReLU)
        
    def forward(self, x):
        snps_enc = self.snp_enc(torch.transpose(x[:, :, :self.num_snps], 1, 2))
        snps_pooled = torch.transpose(self.pool_ind(snps_enc), 1, 2)
        snps_pooled = torch.squeeze(self.pool_features(snps_pooled), dim=1)
        self.att_out = self.att_mask(snps_pooled)
        self.att_out.requires_grad_(True)
        snps_out = self.snps_model(self.att_out)
        
        cov_out = self.cov_model(x[:, :, self.num_snps:])
        data_vec = torch.cat((snps_out, cov_out), dim=-1)
        
        raw_out = self.end_model(data_vec)
        
        return raw_out

class GWANNet2(nn.Module):
    """Attention applied to GroupTrain model after the encoding created
    by the CNN layer and Avg Pooling.

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self, grp_size, enc, h, d, out, activation, 
            num_snps, att_model, att_activ, cov_model):
        super(GWANNet2, self).__init__()

        self.cov_model = cov_model
        self.snp_enc = nn.Conv1d(num_snps, enc, 1)
        self.pool_ind = nn.AvgPool1d(grp_size)        
        self.att_mask = att_model(att_activ)
        self.snps_model = nn.Sequential(
            BasicNN(enc, h, d, out, activation),
            nn.ReLU(),
            nn.BatchNorm1d(out))
        self.att_out = None
        self.num_snps = num_snps    

        self.end_model = BasicNN(32, [16, 8], [0.1, 0.1], 2, nn.ReLU)
        
    def forward(self, x):
        snps_enc = self.snp_enc(torch.transpose(x[:, :, :self.num_snps], 1, 2))
        snps_pooled = torch.squeeze(self.pool_ind(snps_enc), dim=-1)
        self.att_out = self.att_mask(snps_pooled)
        self.att_out.requires_grad_(True)
        snps_out = self.snps_model(self.att_out)
        
        cov_out = self.cov_model(x[:, :, self.num_snps:])
        data_vec = torch.cat((snps_out, cov_out), dim=-1)
        
        raw_out = self.end_model(data_vec)
        
        return raw_out

class GWANNet4(nn.Module):
    """Attention applied to GroupTrain model after the encoding created
    by the CNN layer and Avg Pooling.

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self, grp_size, enc, h, d, out, activation, 
            num_snps, att_model, att_activ, cov_model):
        super(GWANNet4, self).__init__()

        self.cov_model = cov_model
        self.snp_enc = nn.Conv1d(num_snps, enc, 1)
        self.pool_ind = nn.AvgPool1d(grp_size)        
        self.att_mask = att_model(att_activ)
        self.pool_features = nn.AdaptiveMaxPool1d(32)
        self.snps_model = nn.Sequential(
            BasicNN(32, h, d, out, activation),
            nn.ReLU(),
            nn.BatchNorm1d(out))
        self.att_out = None
        self.num_snps = num_snps    

        self.end_model = BasicNN(32, [16, 8], [0.1, 0.1], 2, nn.ReLU)
        
    def forward(self, x):
        snps_enc = self.snp_enc(torch.transpose(x[:, :, :self.num_snps], 1, 2))
        snps_pooled = torch.squeeze(self.pool_ind(snps_enc), dim=-1)
        self.att_out = self.att_mask(snps_pooled)
        self.att_out.requires_grad_(True)
        features_pooled = self.pool_features(
            torch.unsqueeze(self.att_out, dim=1))
        snps_out = self.snps_model(torch.squeeze(features_pooled, dim=1))
        
        cov_out = self.cov_model(x[:, :, self.num_snps:])
        data_vec = torch.cat((snps_out, cov_out), dim=-1)
        
        raw_out = self.end_model(data_vec)
        
        return raw_out

class GWANNet5(nn.Module):
    """Attention applied to GroupTrain model after the encoding created
    by the CNN layer and Avg Pooling.

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self, grp_size, enc, h, d, out, activation, 
            num_snps, att_model, att_activ):
        super(GWANNet5, self).__init__()

        self.snp_enc = nn.Conv1d(num_snps, enc, 1)
        self.pool_ind = nn.AvgPool1d(grp_size)        
        self.att_mask = att_model(att_activ)
        self.pool_features = nn.AdaptiveMaxPool1d(32)
        self.snps_model = nn.Sequential(
            BasicNN(32, h, d, out, activation),
            nn.ReLU(),
            nn.BatchNorm1d(out))
        self.att_out = None
        self.num_snps = num_snps    
        # self.cov_layer_norm = nn.LayerNorm(16)
        # self.snps_layer_norm = nn.LayerNorm(16)
        
        # For T2D and AD
        self.end_model = BasicNN(32, [16, 8], [0.1, 0.1], 2, nn.ReLU)
        
        
        # self.end_model = BasicNN(24, [16, 8], [0.1, 0.1], 2, nn.ReLU)
        
    def forward(self, x):
        snps_enc = self.snp_enc(torch.transpose(x[:, :, :self.num_snps], 1, 2))
        snps_pooled = torch.squeeze(self.pool_ind(snps_enc), dim=-1)
        self.att_out = self.att_mask(snps_pooled)
        self.att_out.requires_grad_(True)
        features_pooled = self.pool_features(
            torch.unsqueeze(self.att_out, dim=1))
        snps_out = self.snps_model(torch.squeeze(features_pooled, dim=1))
        # snps_out = self.snps_layer_norm(snps_out)

        cov_out = torch.mean(x[:, :, self.num_snps:], dim=1)
        # cov_out = self.cov_layer_norm(cov_out)

        data_vec = torch.cat((snps_out, cov_out), dim=-1)
        
        raw_out = self.end_model(data_vec)
        
        return raw_out

class GWANNet6(nn.Module):
    """Attention applied to GroupTrain model after the encoding created
    by the CNN layer and Avg Pooling.

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self, grp_size, cov_size, enc, h, d, out, activation, 
            num_snps, att_model, att_activ):
        super(GWANNet6, self).__init__()

        self.snp_enc = nn.Conv1d(num_snps, enc, 1)
        self.pool_ind = nn.AvgPool1d(grp_size)
        self.att_mask = att_model(att_activ)
        self.pool_features = nn.AdaptiveMaxPool1d(32)
        self.snps_model = nn.Sequential(
            BasicNN(32, h, d, out, activation),
            nn.ReLU(),
            nn.BatchNorm1d(out))
        self.att_out = None
        self.num_snps = num_snps    
        self.cov_enc = nn.Sequential(
            GroupTrainNN(10, cov_size, out, [], [], -1, None),
            nn.ReLU(),
            nn.BatchNorm1d(out))
        
        # self.cov_layer_norm = nn.LayerNorm(16)
        # self.snps_layer_norm = nn.LayerNorm(16)
        self.end_model = BasicNN(32, [16, 8], [0.1, 0.1], 2, nn.ReLU)
        
    def forward(self, x):
        snps_enc = self.snp_enc(torch.transpose(x[:, :, :self.num_snps], 1, 2))
        snps_pooled = torch.squeeze(self.pool_ind(snps_enc), dim=-1)
        self.att_out = self.att_mask(snps_pooled)
        self.att_out.requires_grad_(True)
        features_pooled = self.pool_features(
            torch.unsqueeze(self.att_out, dim=1))
        snps_out = self.snps_model(torch.squeeze(features_pooled, dim=1))
        # snps_out = self.snps_layer_norm(snps_out)

        cov_out = self.cov_enc(x[:, :, self.num_snps:])

        data_vec = torch.cat((snps_out, cov_out), dim=-1)
        
        raw_out = self.end_model(data_vec)
        
        return raw_out

class KDNet(nn.Module):
    """Knowledge distillation model - tiny version of GWANNet5

    """
    def __init__(self, grp_size, enc, h, d, out, activation, 
            num_snps, att_model, att_activ):
        super(KDNet, self).__init__()

        self.snp_enc = nn.Conv1d(num_snps, enc, 1)
        self.pool_ind = nn.AvgPool1d(grp_size)        
        self.att_mask = att_model(att_activ)
        self.snps_model = nn.Sequential(
            BasicNN(enc, h, d, out, activation),
            nn.ReLU(),
            nn.BatchNorm1d(out))
        self.att_out = None
        self.num_snps = num_snps    
        
        # For T2D and AD
        self.end_model = BasicNN(32, [8, 4], [0.0, 0.0], 2, nn.ReLU)
        
        
    def forward(self, x):
        snps_enc = self.snp_enc(torch.transpose(x[:, :, :self.num_snps], 1, 2))
        snps_pooled = torch.squeeze(self.pool_ind(snps_enc), dim=-1)
        self.att_out = self.att_mask(snps_pooled)
        self.att_out.requires_grad_(True)
        snps_out = self.snps_model(torch.squeeze(self.att_out, dim=1))
        # snps_out = self.snps_layer_norm(snps_out)

        cov_out = torch.mean(x[:, :, self.num_snps:], dim=1)
        # cov_out = self.cov_layer_norm(cov_out)

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
        # self.end_model = DiCoNN(enc, h, d, out, activation)
        
    def forward(self, x):
        eout = self.grp_enc(torch.transpose(x, 1, 2))
        # self.att_out = self.att_mask(eout)
        # self.att_out.requires_grad_(True)
        # pooled = torch.squeeze(self.pool(self.att_out), dim=-1)
        # raw_out = self.end_model(pooled)
        
        pooled = torch.squeeze(self.pool(eout), dim=-1)
        self.att_out = self.att_mask(pooled)
        self.att_out.requires_grad_(True)
        raw_out = self.end_model(self.att_out)
        
        return raw_out

class GroupBetAttention(nn.Module):
    """Attention applied to GroupTrain model before applying
    traditional CNN layer and Avg Pooling. The attention is sort a CNN
    itself.

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self, grp_size, inp, enc, h, d, out, activation, beta_matrix):
        super(GroupBetAttention, self).__init__()

        self.beta_matrix = beta_matrix
        self.beta_matrix.requires_grad_(True)
        self.beta_regression = nn.Conv1d(6, 1, 1)
        self.grp_enc = nn.Conv1d(inp, enc, 1)
        self.pool = nn.AvgPool1d(grp_size)
        self.end_model = BasicNN(enc, h, d, out, activation)
        
    def forward(self, x):
        beta_att = self.beta_regression(self.beta_matrix)
        beta_att = torch.squeeze(beta_att)
        beta_att = beta_att.repeat(x.shape[0]*x.shape[1], 1)
        beta_att = beta_att.reshape((x.shape[0], x.shape[1], -1))
        cov_shape = list(beta_att.shape)
        cov_shape[-1] = x.shape[-1] - cov_shape[-1]
        beta_att = torch.cat((beta_att, torch.ones(cov_shape, device=x.device)), 
            dim=-1)
        x = x*beta_att
        x = torch.transpose(x, 1, 2)
        eout = self.grp_enc(x)
        pooled = torch.squeeze(self.pool(eout), dim=-1)
        raw_out = self.end_model(pooled)

        return raw_out

class GroupTrainNN(nn.Module):
    def __init__(self, grp_size, inp, enc, h, d, out, activation):
        super(GroupTrainNN, self).__init__()

        self.grp_enc = nn.Conv1d(inp, enc, 1)
        self.pool = nn.AvgPool1d(grp_size)
        # self.end_model = BasicNN(enc, h, d, out, activation)
        # self.end_model = DiCoNN(enc, h, d, out, activation)
        
    def forward(self, x):
        
        eout = self.grp_enc(torch.transpose(x, 1, 2))
        pooled = torch.squeeze(self.pool(eout), dim=-1)
        # pooled.requires_grad_(True)
        # raw_out = self.end_model(pooled)
        raw_out = pooled
        return raw_out #, pooled

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
        for l, drop, bnorm, activ in zip(self.linears[1:], self.dropouts, self.bnorms,
                                        self.activation):
            
            # Different batch norm, dropout and relu combinations
            # X = l(drop(activ(bnorm(X))))
            # X = l(activ(bnorm(X)))
            # X = l(bnorm(activ(X)))
            # X = l(drop(activ(X)))
            X = l(bnorm(activ(drop(X))))
            
        return X

class DiCoNN(nn.Module):
    def __init__(self, inp, h, d, out, activation, num_snps, inp_limit=200, 
                    encoding_size=32):
        super(DiCoNN, self).__init__()

        self.num_snps = num_snps
        self.num_cov = inp - num_snps
        self.snp_limit = inp_limit - self.num_cov
        self.encoding_size = encoding_size
        num_parallel_models, rem = divmod(num_snps, self.snp_limit)
        self.nn_list = nn.ModuleList()
        for _ in range(num_parallel_models):
            m = BasicNN(inp_limit, h, d, encoding_size, activation)
            self.nn_list.append(m)
        if rem:
            m = BasicNN(rem+self.num_cov, h, d, encoding_size, activation)
            self.nn_list.append(m)

        self.activs = [activation() for m in self.nn_list]
        self.final_layer = nn.Linear(len(self.nn_list)*self.encoding_size, out)
        
    def forward(self, X):
        encodings = []
        for idx, m in enumerate(self.nn_list):
            if idx == len(self.nn_list)-1:
                inp = torch.cat((X[:,idx*self.snp_limit:self.num_snps], X[:, -self.num_cov:]), dim=1)
            else:
                inp = torch.cat((X[:,idx*self.snp_limit:(idx+1)*self.snp_limit], X[:, -self.num_cov:]), dim=1)
            enc = self.activs[idx](m.forward(inp))
            encodings.append(enc)
        raw_out = self.final_layer(torch.cat(encodings, dim=1))
        return raw_out

class SimpleLSTM(nn.Module):

    def __init__(self, inp, hunits, hlayers, out):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(inp, hunits, hlayers, batch_first=True)
        self.output_layer = nn.Linear(hunits, out)

    def forward(self, X):

        lstm_out, (h_n, c_n) = self.lstm(X)
        raw_out = self.output_layer(h_n[-1])

        return raw_out

class SNPEncoder(nn.Module):
    
    def __init__(self, inp, hunits, hlayers, drop, bi=True):
        super(SNPEncoder, self).__init__()
        
        self.inp = inp
        self.bilstm = nn.LSTM(input_size=inp, hidden_size=hunits, 
            num_layers=hlayers, batch_first=True, dropout=drop, 
            bidirectional=bi)
    
    def forward(self, x):
        raw_out, _ = self.bilstm(x)
        # h).view(seq_len, batch, num_directions, hidden_size)
        return raw_out[:, -1]

class SNPDecoder(nn.Module):
    def __init__(self, inp, hunits, hlayers, out, drop, bi=True):
        super(SNPDecoder, self).__init__()
        
        self.bilstm = nn.LSTM(inp, hunits, hlayers, batch_first=True, 
            dropout=drop, bidirectional=bi)
        self.output = nn.Linear(hunits*2*bi, out)
    
    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        raw_out = self.output(lstm_out[:,-1])
        
        return raw_out

class EnDeNN(nn.Module):
    def __init__(self, enc, dec, num_snps):
        super(EnDeNN, self).__init__()
        
        self.encoder = enc
        self.decoder = dec
        self.num_snps = num_snps
    
    def forward(self, x):
        
        one_hot = F.one_hot(x[:, :self.num_snps].long(), self.encoder.inp).float()
        enc_out = self.encoder(one_hot)
        
        dec_inp = torch.cat((enc_out, x[:, self.num_snps:]), dim=1)
        # dec_inp = torch.unsqueeze(dec_inp, -1)
        dec_out = self.decoder(dec_inp)
        
        return dec_out