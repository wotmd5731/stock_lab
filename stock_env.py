# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 23:30:48 2019

@author: JAE
"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class env_stock():
    def __init__(self,count_max,view_seq,init_money):
        
        self.count_max= count_max
        self.view_seq = view_seq
        self.init_money = init_money
        
    def reset(self):
        stock_code = np.random.choice(['005930' , '000270','000660','005380','005490','009240','009540'])
        self.start_date = '2018-01-01'
        self.end_date = '2019-01-01'
        
        
        self.prev_action = 0
        self.count = 0
        self.balance = self.init_money
        self.num_stocks = 0
        self.sum_action = 0
        
        chart_data = data_manager.load_chart_data(
            os.path.join('./',
                         'data/chart_data/{}.csv'.format(stock_code)))
        prep_data = data_manager.preprocess(chart_data)
        training_data = data_manager.build_training_data(prep_data)
        
        # 기간 필터링
        start = random.randint(self.view_seq ,(len(training_data)-self.count_max-200))
        
        training_data = training_data[start-self.view_seq :start+self.count_max+200]
        
#        training_data = training_data[(training_data['date'] >= self.start_date) &
#                                      (training_data['date'] <= self.end_date)]
        training_data = training_data.dropna()
        
        
        # 차트 데이터 분리
        features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
        self.chart_data = training_data[features_chart_data]
        self.chart_data['date']= pd.to_datetime(self.chart_data.date).astype(np.int64)/1e12


        # 학습 데이터 분리
        features_training_data = [
            'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
            'close_lastclose_ratio', 'volume_lastvolume_ratio',
            'close_ma5_ratio', 'volume_ma5_ratio',
            'close_ma10_ratio', 'volume_ma10_ratio',
            'close_ma20_ratio', 'volume_ma20_ratio',
            'close_ma60_ratio', 'volume_ma60_ratio',
            'close_ma120_ratio', 'volume_ma120_ratio'
        ]
        training_data = training_data[features_training_data]
    

        
        self.data = torch.from_numpy(training_data.values).float()


        
        state = self.data[self.count :self.count+self.view_seq].view(1,-1)
        state = torch.cat([state, torch.Tensor([self.sum_action]).view(1,-1)],dim=1)
        return state
        pass
        

    def step(self,action):
        #continuouse action space
        r_t = d_t= 0
        action = action.item()
        self.sum_action = np.clip(self.sum_action + action, 0,100)
        aa = round(self.sum_action)
        
        delta_a = (self.prev_action - aa)
        
        self.prev_action = aa
        
        
        cost = self.data[self.count,1]*delta_a*0.99
        self.pocket += cost
        
#        r_t = cost
#        r_t = cost
        if self.count+1 == self.count_max:
#            self.pocket += self.data[self.count,1]*(self.prev_action*quantize)
            d_t = 1
            r_t = self.pocket 

        else:
            self.count +=1


        
        state = self.data[self.count :self.count+self.view_seq].view(1,-1)
        state = torch.cat([state, torch.Tensor([self.sum_action]).view(1,-1)],dim=1)
        
        return state ,r_t ,d_t,0
        pass
    
    def vis(self):
        self.win1= vis.line(Y=self.data[:,1],win=self.win1,opts=dict(title='price'))
        self.win2= vis.line(Y=self.data[:,2],win=self.win2,opts=dict(title='vol'))
        





# In[25]:



STEP_MAX =400
WINDOW_SIZE = 2


env = env_stock(STEP_MAX,WINDOW_SIZE,10000000)
env.reset()