# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 23:30:48 2019

@author: JAE
"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch


#%%

class env_stock():
    def __init__(self,count_max,init_money,period):
        self.code_list = ['005930.KS' , '000270.KS','000660.KS','005380.KS','005490.KS','009240.KS','009540.KS']
        self.period = period 
        
        self.count_max= count_max
#        self.view_seq = view_seq
        self.init_money = init_money
        self.min_date ='2015-01-01'
        self.max_date ='2016-01-01'
        self.window_size = 10
        
        
    def reset(self):
        stock_code = np.random.choice(self.code_list)
        
        s = pd.date_range(self.min_date,self.max_date,freq="D")
        sample_idx = random.randrange(self.window_size,len(s)-self.period)
        start_date = str(s[sample_idx-self.window_size]).split()[0]
        end_date = str(s[sample_idx+self.period]).split()[0]
        
        self.src_data = data.DataReader(stock_code, 'yahoo', start_date, end_date)

        #프리프로세싱
        #data = source['Close']
        sdata = self.src_data
        
        #ret = data.pct_change(1)
        log_ret = np.log(sdata/sdata.shift(1))
        log_ret.columns = ['log_h','log_l','log_o','log_c','log_v','log_adj']
        
        
        def min_max_norm(wdata):
            return ( wdata[-1] - wdata.min() )/ (wdata.max()-wdata.min())
        def mean_std_norm(wdata):
            return ( wdata[-1] - wdata.mean() )/ wdata.std()
        
        
        def svd_whiten(X):
        #    a = source[X]
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            # U and Vt are the singular matrices, and s contains the singular values.
            # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
            # will be white
            X_white = np.dot(U, Vt)
            return X_white[-1]
        
        def rolling_whiten(src,window=20,min_periods=10):
            ret = []
            for i in range(src.__len__()):
                if i < min_periods-1:
                    ret.append([0 for i in range(src.columns.__len__())])
                elif i<window:
                    ret.append(svd_whiten(src[0:i].values))
                else:
                    ret.append(svd_whiten(src[i-window:i].values))
            pdata = pd.DataFrame(np.stack(ret,0))
            pdata.columns = ['w'+str(i) for i in range(src.columns.__len__())]
            pdata.index = src.index
            return pdata
        
        
        
        norm0 = sdata.rolling(window=self.window_size,min_periods=10).apply(min_max_norm,raw=True).fillna(0)
        
        norm1 = sdata.rolling(window=self.window_size,min_periods=10).apply(mean_std_norm,raw=True).fillna(0)
        norm2 = rolling_whiten(self.src_data,window=self.window_size,min_periods=10).fillna(0)#pca whitening using close open min max vol
        
        norm0.columns = ['mm_h','mm_l','mm_o','mm_c','mm_v','mm_adj']
        norm1.columns = ['ms_h','ms_l','ms_o','ms_c','ms_v','ms_adj']


        self.prep_data = pd.concat([self.src_data, norm0,norm1,norm2,log_ret],axis=1)


        self.prev_action = 0
        self.count = self.window_size
        self.balance = self.init_money
        self.num_stocks = 0
        self.sum_action = 0
        
        
        
        

        # 학습 데이터 분리
#        features_training_data = [
#            'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
#            'close_lastclose_ratio', 'volume_lastvolume_ratio',
#            'close_ma5_ratio', 'volume_ma5_ratio',
#            'close_ma10_ratio', 'volume_ma10_ratio',
#            'close_ma20_ratio', 'volume_ma20_ratio',
#            'close_ma60_ratio', 'volume_ma60_ratio',
#            'close_ma120_ratio', 'volume_ma120_ratio'
#        ]
    
        da0 = self.prep_data.iloc[self.count]
        
        state = torch.from_numpy(da0.values).float()
        
#        state = torch.cat([state, torch.Tensor([self.sum_action]).view(1,-1)],dim=1)
        return state
        

    def step(self,action):
        #continuouse action space
        r_t = d_t= 0
        action = action.item()
        self.sum_action = np.clip(self.sum_action + action, 0,100)
        aa = round(self.sum_action)
        
        delta_a = (self.prev_action - aa)
        
        self.prev_action = aa
        
        
        cost = self.prep_data.iloc[self.count]['Close']*delta_a*0.99
        self.pocket += cost
        
#        r_t = cost
#        r_t = cost
        r_t = self.pocket 
        self.count +=1
        if self.count+1 == self.count_max:
#            self.pocket += self.data[self.count,1]*(self.prev_action*quantize)
            d_t = 1
        

        #calc  new state
        da0 = self.prep_data.iloc[self.count]
        state = torch.from_numpy(da0.values).float()
        
        return state ,r_t ,d_t,0
        pass
    
    def vis(self):
        self.win1= vis.line(Y=self.data[:,1],win=self.win1,opts=dict(title='price'))
        self.win2= vis.line(Y=self.data[:,2],win=self.win2,opts=dict(title='vol'))
        





# In[25]:




env = env_stock(400,100000,10)
env.reset()

