import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, GRU
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    NNConv,
    DiffGroupNorm
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
from torch.autograd import Variable
import numpy as np

def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #print(2 * torch.log(sig_p / sig_q).sum())
    #print(((sig_q / sig_p).pow(2)).sum())
    #print((((mu_p - mu_q) / sig_p).pow(2)).sum())
    return KLD


class BayesLinear(torch.nn.Module):
    
    """
    Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
    the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
    with gaussian priors.
    """
    
    def __init__(self, n_in, n_out, prior_sig = 0.15, bias = True):
        
        super(BayesLinear, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.prior_sig = prior_sig
        
        self.W_p = None
        self.b_p = None
        
        # initialise mu for weights
        self.W_mu = torch.nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.02, 0.02))
        #self.W_mu = self.W_mu.to("cuda")
        # initialise mu for biases
        if self.bias:
            self.b_mu = torch.nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.02, 0.02))

    
    def init_rho(self, p_min, p_max):
        self.W_p = torch.nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(p_min, p_max))
        if self.bias:
            self.b_p = torch.nn.Parameter(torch.Tensor(self.n_out).uniform_(p_min, p_max))     


    def forward(self, X, sample=False):
        
        if sample:

            ### weights
            
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20) # compute stds for weights
            act_W_mu = torch.mm(X, self.W_mu)  # activation means
            act_W_std = torch.sqrt(torch.clamp_min(torch.mm(X.pow(2), std_w.pow(2)),1e-6)) # actiavtion stds
            
            eps_W = Variable(self.W_mu.data.new(act_W_std.size()).normal_(mean=0, std=1)) # draw samples from 0,1 gaussian
            act_W_out = act_W_mu + act_W_std * eps_W # sample weights from 'posterior'
            
            output = act_W_out
            kld = KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.W_mu, sig_q=std_w)
            
            if self.bias:
                
                std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20) # compute stds for biases
                eps_b = Variable(self.b_mu.data.new(std_b.size()).normal_(mean=0, std=1)) # draw samples from 0,1 gaussian
                act_b_out = self.b_mu + std_b * eps_b # sample biases from 'posterior'
                output += act_b_out.unsqueeze(0).expand(X.shape[0], -1)
                kld += KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.b_mu, sig_q=std_b)
       
            return output, kld
    #kld
        
        
        else:
            
            output = torch.mm(X, self.W_mu)
            
            # kld is just standard regularisation term
            kld = (0.5*((self.W_mu / self.prior_sig).pow(2))+torch.log(self.prior_sig*torch.sqrt(torch.tensor(2*np.pi)))).sum()
            
            if self.bias:
                output += self.b_mu.expand(X.size()[0], self.n_out)
                
                kld += (0.5*((self.b_mu / self.prior_sig).pow(2))+torch.log(self.prior_sig*torch.sqrt(torch.tensor(2*np.pi)))).sum()
            
            
            return output, kld
       
    
# CGCNN
class MPNN_Bayes(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        dim3=64,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        prior_sig = 0.15,
        sample = False,
        **kwargs
    ):
        super(MPNN_Bayes, self).__init__()

        
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.prior_sig = prior_sig
        self.sample = sample
        
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"        
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1
        ##Determine output dimension length
        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = BayesLinear(data.num_features, dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = BayesLinear(dim1, dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        ##Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.gru_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            nn = Sequential(
                Linear(data.num_edge_features, dim3), ReLU(), Linear(dim3, gc_dim * gc_dim)
            )
            conv = NNConv(
                gc_dim, gc_dim, nn, aggr="mean"
            )            
            self.conv_list.append(conv)
            gru = GRU(gc_dim, gc_dim)
            self.gru_list.append(gru)

            ##Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm == "True":
                #bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
                bn = DiffGroupNorm(gc_dim, 10, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        ##Set up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = BayesLinear(post_fc_dim * 2, dim2)
                    else:
                        lin = BayesLinear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = BayesLinear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = BayesLinear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = BayesLinear(post_fc_dim*2, output_dim)
            else:
                self.lin_out = BayesLinear(post_fc_dim, output_dim)   

        ##Set up set2set pooling (if used)
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = BayesLinear(output_dim * 2, output_dim)
    
        self.create_log_noise(1)
            
            
    def create_log_noise(self, num):
        self.log_noise = torch.nn.Parameter(torch.ones(num))
        
    def forward(self, data, sample = False):
        
        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out,tkl = self.pre_lin_list[i](data.x, sample)
                out = getattr(F, self.act)(out)
                #prev_out = out
            else:
                out,tkl = self.pre_lin_list[i](out, sample)
                out = getattr(F, self.act)(out)
                #out = torch.add(out, prev_out)
                #prev_out = out
        prev_out = out

        ##GNN layers
        if len(self.pre_lin_list) == 0:
            h = data.x.unsqueeze(0)    
        else:
            h = out.unsqueeze(0)                
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    m = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                    m = self.bn_list[i](m)
                else:
                    m = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
            else:
                if self.batch_norm == "True":
                    m = self.conv_list[i](out, data.edge_index, data.edge_attr)
                    m = self.bn_list[i](m)
                else:
                    m = self.conv_list[i](out, data.edge_index, data.edge_attr)            
            m = getattr(F, self.act)(m)          
            m = F.dropout(m, p=self.dropout_rate, training=self.training)
            out, h = self.gru_list[i](m.unsqueeze(0), h)
            out = out.squeeze(0)
            out = torch.add(out, prev_out)
            prev_out = out            

        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
            for i in range(0, len(self.post_lin_list)):
                out, kl = self.post_lin_list[i](out, sample)
                tkl += kl
                out = getattr(F, self.act)(out)
                #out = torch.add(out, prev_out)
                #prev_out = out
            out,kl = self.lin_out(out, sample)
            tkl += kl
            #out = torch.add(out, prev_out)
            #prev_out = out

        elif self.pool_order == "late":
            for i in range(0, len(self.post_lin_list)):
                out, kl = self.post_lin_list[i](out, sample)
                tkl += kl
                out = getattr(F, self.act)(out)
                #out = torch.add(out, prev_out)
                #prev_out = out
            out,kl = self.lin_out(out)
            tkl += kl
            #out = torch.add(out, prev_out)
            #prev_out = out

            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out,kl = self.lin_out_2(out, sample)
                tkl += kl
                #out = torch.add(out, prev_out)
                #prev_out = out
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                
        if out.shape[1] == 1:
            return out.view(-1), tkl
        else:
            return out, tkl
        
        
        
        
        
        
        
        
        
        
        