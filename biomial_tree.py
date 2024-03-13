"""
Binary tree model
"""

import numpy as np
import matplotlib.pyplot as plt

class Binomial_Tree():
    """
    Binary_tree class.
    Inputs:
        -   S: stock price
        -   r: risk-free interest rate
        -   vol: volatility % in decimals
        -   T: Time period
        -   N: Number of steps/intervals
        -   auto: Compute tree automatically, True as default
    """

    def __init__(self,S,r,vol,T,N, K,auto = True):
        
        self.S = S
        self.r = r
        self.vol = vol
        self.T = T
        self.N = N
        self.dt = T/N
        
        if auto == True:
            self.build_tree()
            self.valueOptionMatrix(K)

    def build_tree(self):
    
        matrix = np.zeros((self.N+1,self.N+1))
        
        u = np.exp(self.vol*np.sqrt(self.dt))
        d = np.exp(-self.vol*np.sqrt(self.dt))
        matrix[0,0] = self.S
        
        
        for i in np.arange(self.N+1) :
            for j in np.arange(i+1) : 

                matrix[i,j] = self.S*(u**(j))*(d**(i-j))
        
        self.tree = matrix



    def valueOptionMatrix(self,K):

        self.K = K

        columns = self.tree.shape[1]
        rows = self.tree.shape[0]
        v_tree_european = np.copy(self.tree)
        v_tree_european_put = np.copy(self.tree)
        v_tree_american = np.copy(self.tree)
        v_tree_american_put = np.copy(self.tree)


        u= np.exp(self.vol*np.sqrt(self.dt))

        d= np.exp(-self.vol*np.sqrt(self.dt))

        p= (np.exp(self.r*self.dt) - d)/(u-d)   

        for c in np.arange(columns):
            St = self.tree[rows - 1, c] 
            v_tree_european[rows - 1, c] = max(0.,  St - self.K)
            v_tree_european_put[rows -1,c] = max(0., self.K - St)
            v_tree_american[rows -1,c] = max(0., St-self.K)
            v_tree_american_put[rows -1,c] = max(0., self.K - St)

        for i in np.arange(rows - 1)[:: -1]:
            for j in np.arange(i + 1):
                european_down = v_tree_european[ i + 1, j ]
                european_up = v_tree_european[ i + 1, j + 1]

                european_down_put = v_tree_european_put[ i + 1, j ]
                european_up_put = v_tree_european_put[ i + 1, j + 1]

                american_down= v_tree_american[ i + 1, j ]
                american_up= v_tree_american[ i + 1, j + 1]

                american_down_put = v_tree_american_put[i + 1, j ]
                american_up_put = v_tree_american_put[i + 1, j +1]

                
                v_tree_european[i , j ] = np.exp(-self.r*self.dt)*(p*european_up + (1-p)*european_down)

                v_tree_european_put[i , j ] = np.exp(-self.r*self.dt)*(p*european_up_put + (1-p)*european_down_put)
                
                v_tree_american[i,j] = max(np.exp(-self.r*self.dt)*(p*american_up + (1-p)*american_down), self.tree[i,j]-self.K)

                v_tree_american_put[i,j] = max(np.exp(-self.r*self.dt)*(p*american_up_put + (1-p)*american_down_put), self.K - self.tree[i,j])
       
        self.v_tree_european = v_tree_european
        self.v_tree_european_put = v_tree_european_put
        self.v_tree_american = v_tree_american
        self.v_tree_american_put = v_tree_american_put
        self.delta = (v_tree_european[1,1] - v_tree_european[1,0])/(self.S*u - self.S*d)

        
