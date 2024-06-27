import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    '''
    Binary Ising spins RBM with one hidden layer and one visible layer. 
    '''
    def __init__(self, n_visible=None, n_hidden=None, v_bias=None, h_bias=None, J=None):
        super().__init__()
        
        self.v_bias = nn.Parameter(
            v_bias if v_bias is not None else 
            torch.randn(1, n_visible)
        )
        
        self.h_bias = nn.Parameter(
            h_bias if h_bias is not None else 
            torch.zeros(1, n_hidden) - 1
        )
        
        self.W = nn.Parameter(
            J if J is not None else
            torch.randn(n_visible, n_hidden)
        )
        
    def hamiltonian(self, v, h):
        '''
        Returns the Hamiltonian energy of the RBM
        '''
        return -(
            F.bilinear(v, h, self.W.unsqueeze(0)) + 
            F.linear(self.v_bias, v) + 
            F.linear(self.h_bias, h)
        )
        
    def h_given_v(self, v):
        '''
        Returns the conditional distribution P(h | v)
        '''
        return torch.sigmoid(F.linear(v, self.W.t(), self.h_bias))
    
    def v_given_h(self, h):
        '''
        Returns the conditional distribution P(v | h)
        '''
        return torch.sigmoid(F.linear(h, self.W, self.v_bias))
    
    def forward(self, v, k=1):
        '''
        Returns the distribution over visible spins after k Gibbs sampling steps. 
        '''
        for i in range(k):
            h = self.h_given_v(v.bernoulli())
            v = self.v_given_h(h.bernoulli())
        
        return v
