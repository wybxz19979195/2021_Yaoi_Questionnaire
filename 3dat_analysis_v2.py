# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 09:07:22 2021

@author: Administrator
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import palettable
def t_test(dt):    
    D =  pd.DataFrame(np.empty([dt.shape[1]-3,6]),columns = ['p','t','yaoi_mean','non_yaoi_mean','yaoi_std','non_yaoi_std'])   
    for i in range(dt.shape[1]-3):
        subject1 = dt[(dt['143、您看耽美吗']==1) & (dt['D_total']>2)].iloc[:,i+3]      
        subject2 = dt[(dt['143、您看耽美吗']==2) & (dt['D_total'] < 3)].iloc[:,i+3]
        r = stats.ttest_ind(subject1,subject2)
        D.iloc[i,1] = r.__getattribute__("statistic")
        D.iloc[i,0] = r.__getattribute__("pvalue")
        D.iloc[i,2] = subject1.mean()
        D.iloc[i,3] = subject2.mean()
        D.iloc[i,4] = subject1.std()
        D.iloc[i,5] = subject2.std()
    D.index = list(dt.columns.values[3:dt.shape[1]])
    return D


def cor_plot(dt,date,name  ):    
    sns.heatmap(data=dt.corr(method='pearson'),
    vmax=0.3,
    cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
    annot=True,
    fmt=".2f",
    annot_kws={'size':8,'weight':'normal', 'color':'#253D24'},
    mask=np.triu(np.ones_like(dt.corr(method='pearson'),dtype=np.bool)),#显示对脚线下面部分图
    square=True, linewidths=.5,#每个方格外框显示，外框宽度设置
    cbar_kws={"shrink": .5})
    plt.savefig(os.path.join('result',date+'_' + name+'.tiff'),dpi=100)
    plt.close()

def plot_gaussion(dt,p_name,date):    
    s = dt
    #画散点图和直方图
    fig = plt.figure(figsize = (10,6))
    ax1 = fig.add_subplot(2,1,1) # 创建子图1
    ax1.scatter(s.index, s.values)
    plt.grid()
    ax2 = fig.add_subplot(2,1,2) # 创建子图2
    s.hist(bins=30,alpha = 0.5,ax = ax2)
    s.plot(kind = 'kde', secondary_y=True,ax = ax2)
    plt.title(p_name)
    plt.grid()
    plt.show()
    plt.savefig(os.path.join('result',date+'_' +p_name+'.tiff'),dpi=100)
    plt.close()
    u = s.mean() # 计算均值
    std = s.std() # 计算标准差
    stats.kstest(s, 'norm', (u, std))
    
date = '1003_v3_' 

dt_dis = pd.read_excel(date+'3_distinguish_new2.xlsx')  
plot_gaussion(dt_dis[ (dt_dis['143、您看耽美吗']==1) & (dt_dis['D_total']>2) & (dt_dis['time_ln_z']> -1.5) ]['D_total'],'fans_distinguish_total',date)
plot_gaussion(dt_dis[ (dt_dis['143、您看耽美吗']==2) & (dt_dis['D_total'] < 3)& (dt_dis['time_ln_z']> -1.5)]['D_total'],'non_fans_distinguish_total',date)
plot_gaussion(dt_dis[dt_dis['time_ln_z']> -1.5]['D_total'],'total_fans_distinguish_total',date)
  
dt_bf = pd.read_excel(date+'4_big_five.xlsx')  
dt_bf_useful_yaoi = dt_bf[ (dt_bf['143、您看耽美吗']==1) & (dt_bf['D_total']>2) & (dt_dis['time_ln_z']> -1.5)]
dt_bf_useful_yaoi_non =dt_bf[ (dt_bf['143、您看耽美吗']==2) & (dt_bf['D_total'] < 3 ) & (dt_dis['time_ln_z']> -1.5)]
dt_bf_useful = pd.concat([dt_bf_useful_yaoi,dt_bf_useful_yaoi_non])
dt_bf_useful.to_excel(date+'4_big_five'+'_useful.xlsx')
plot_gaussion(dt_bf_useful_yaoi['P_total'],'fans_'+'big_five',date)
plot_gaussion(dt_bf_useful_yaoi_non['P_total'],'non_fans_'+'big_five',date)
plot_gaussion(pd.concat([dt_bf_useful_yaoi['P_total'],dt_bf_useful_yaoi_non['P_total']]),'total_fans_'+'big_five',date)
cor_plot(dt_bf_useful.iloc[:,1:dt_bf.shape[1]],date,'cor_'+'+big_five') 
cor_plot(dt_bf_useful_yaoi.iloc[:,1:dt_bf.shape[1]],date,'cor_'+'big_five_yaoi')  
cor_plot(dt_bf_useful_yaoi_non.iloc[:,1:dt_bf.shape[1]],date,'cor_'+'big_five_yaoi_non') 
t_bf = t_test(dt_bf_useful )  
    

q_name = '4_big_five_divided'
data = pd.read_excel(date+ q_name+'.xlsx')  
useful_yaoi = data[ (data['143、您看耽美吗']==1) & (data['D_total']>2) & (dt_dis['time_ln_z']> -1.5)]
useful_yaoi_non = data[ (data['143、您看耽美吗']==2) & (data['D_total'] <3) & (dt_dis['time_ln_z']> -1.5)]
useful = pd.concat([useful_yaoi,useful_yaoi_non])
useful.to_excel(date+ q_name +'_useful.xlsx')
cor_plot(useful.iloc[:,1:data.shape[1]],date,'cor_'+ q_name) 
cor_plot(useful_yaoi.iloc[:,1:data.shape[1]],date,'cor_'+q_name+'_yaoi')  
cor_plot(useful_yaoi_non.iloc[:,1:data.shape[1]],date,'cor_'+q_name+'_yaoi_non') 
t_bf_di = t_test(useful )  


q_name = '5_attitude_homosexual'
data = pd.read_excel(date+ q_name+'.xlsx')  
useful_yaoi = data[ (data['143、您看耽美吗']==1) & (data['D_total']>2) & (dt_dis['time_ln_z']> -1.5)]
useful_yaoi_non = data[ (data['143、您看耽美吗']==2) & (data['D_total'] <3) & (dt_dis['time_ln_z']> -1.5)]
useful = pd.concat([useful_yaoi,useful_yaoi_non])
useful.to_excel(date+ q_name +'_useful.xlsx')
plot_gaussion(useful_yaoi['AT_total'],'fans_'+q_name,date)
plot_gaussion(useful_yaoi_non['AT_total'],'non_fans_'+q_name,date)
plot_gaussion(pd.concat([useful_yaoi['AT_total'],useful_yaoi_non['AT_total']]),'total_fans_'+ q_name,date)
cor_plot(useful.iloc[:,1:data.shape[1]],date,'cor_'+ q_name) 
cor_plot(useful_yaoi.iloc[:,1:data.shape[1]],date,'cor_'+q_name+'_yaoi')  
cor_plot(useful_yaoi_non.iloc[:,1:data.shape[1]],date,'cor_'+q_name+'_yaoi_non') 
t_AT = t_test(useful )  

q_name = '6_emotion_creativity'
data = pd.read_excel(date+ q_name+'.xlsx')  
useful_yaoi = data[ (data['143、您看耽美吗']==1) & (data['D_total']>2) & (dt_dis['time_ln_z']> -1.5)]
useful_yaoi_non = data[ (data['143、您看耽美吗']==2) & (data['D_total'] <3) & (dt_dis['time_ln_z']> -1.5)]
useful = pd.concat([useful_yaoi,useful_yaoi_non])
useful.to_excel(date+ q_name +'_useful.xlsx')
plot_gaussion(useful_yaoi['ECI_total'],'fans_'+q_name,date)
plot_gaussion(useful_yaoi_non['ECI_total'],'non_fans_'+q_name,date)
plot_gaussion(pd.concat([useful_yaoi['ECI_total'],useful_yaoi_non['ECI_total']]),'total_fans_'+ q_name,date)
cor_plot(useful.iloc[:,1:data.shape[1]],date,'cor_'+ q_name) 
cor_plot(useful_yaoi.iloc[:,1:data.shape[1]],date,'cor_'+q_name+'_yaoi')  
cor_plot(useful_yaoi_non.iloc[:,1:data.shape[1]],date,'cor_'+q_name+'_yaoi_non') 
t_ECI = t_test(useful )  
t = pd.concat([t_AT,t_bf,t_bf_di,t_ECI])
t.to_excel(str(date)+'all_useful_t_test.xlsx')

q_name = '7_motive_yaoi'
data = pd.read_excel(date+ q_name+'.xlsx')  
useful_yaoi = data[ (data['143、您看耽美吗']==1) & (data['D_total']>2) & (dt_dis['time_ln_z']> -1.5)]
useful_yaoi.to_excel(date+ q_name +'yaoi_useful.xlsx')
plot_gaussion(useful_yaoi['M_total'],'fans_'+q_name,date)
cor_plot(useful_yaoi.iloc[:,1:data.shape[1]],date,'cor_'+q_name+'_yaoi')  

q_name = '8_addict_yaoi'
data = pd.read_excel(date+ q_name+'.xlsx')  
useful_yaoi = data[ (data['143、您看耽美吗']==1) & (data['D_total']>2) & (dt_dis['time_ln_z']> -1.5)]
useful_yaoi.to_excel(date+ q_name +'yaoi_useful.xlsx')
plot_gaussion(useful_yaoi['A_total'],'fans_'+q_name,date)
cor_plot(useful_yaoi.iloc[:,1:data.shape[1]],date,'cor_'+q_name+'_yaoi')  



number_yaoi_use = dt_bf[ dt_bf['143、您看耽美吗']==1]
number_useful_yaoi_time = dt_bf[ (dt_bf['143、您看耽美吗']==1) & (dt_dis['time_ln_z']> -1.5)]
number_useful_yaoi_dis = dt_bf[ (dt_bf['143、您看耽美吗']==1) & (dt_bf['D_total']>2)]
number_yaoi_use.shape[0] - number_useful_yaoi_time.shape[0] # delete ac time
number_yaoi_use.shape[0] - number_useful_yaoi_dis.shape[0] # delete ac dis
number_yaoi_use.shape[0] - useful_yaoi.shape[0] # total

number_yaoi_non_use = dt_bf[ dt_bf['143、您看耽美吗']==2]
number_useful_yaoi_non_time = dt_bf[ (dt_bf['143、您看耽美吗']==2) & (dt_dis['time_ln_z']> -1.5)]
number_useful_yaoi_non_dis = dt_bf[ (dt_bf['143、您看耽美吗']==2) & (dt_bf['D_total'] < 3)]
number_yaoi_non_use.shape[0] - number_useful_yaoi_non_time.shape[0] # delete ac time
number_yaoi_non_use.shape[0] - number_useful_yaoi_non_dis.shape[0] # delete ac dis
number_yaoi_non_use.shape[0] - dt_bf_useful_yaoi_non.shape[0] # total
