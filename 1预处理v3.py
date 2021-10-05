# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 16:02:37 2021

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

def Reverse_score(datas,r_item):  #反向记分
    for event in r_item:
        for order in range(0,len(datas)):
            if datas[event][order]==1:
                datas[event][order]=5
            elif  datas[event][order]==2:
                datas[event][order]=4
            elif  datas[event][order]==4:
                datas[event][order]=2
            elif  datas[event][order]==5:
                datas[event][order]=1
    return(datas)

def dimension_grade(name,d_dt,d_name):  #计算维度总分  name维度名称；d_dt:数据；d_name:维度包含题目list
    D =  pd.DataFrame(np.empty([d_dt.shape[0],len(name)]),columns = name) 
    for d in range(len(name)):  
        print(d)        
        D.iloc[:,[d]] = pd.DataFrame( pd.concat( [pd.DataFrame([d_dt[d_name[d][i]] 
                                                 for  i in range(len(d_name[d])) ]).T.apply(lambda x: x.sum(),axis=1)] ) ) 
    return D




date = '1001_v2'   #日期+第几批转账
file = pd.read_excel( date+'_1522'+'.xlsx')#  被试编号
dt = file.copy()

dt_yaoi = dt[dt['143、您看耽美吗']==1]
dt_non_yaoi =  dt[dt['143、您看耽美吗']==2]

dt_lie_item = pd.DataFrame(np.empty([dt.shape[0],5]),columns = ['51、我从来没有遇到过让我难过的事','102、本题请选择最左边的选项',
                                                   '153、本题请选择中间的选项','204、我从来没有撒过谎','250、本题请选择最右边的选项'])#构建表
for i  in  range(dt.shape[0]):
    if dt['250、本题请选择最右边的选项'][i] == 5:
       dt_lie_item ['250、本题请选择最右边的选项'][i] = 1
    else:     
         dt_lie_item ['250、本题请选择最右边的选项'][i] = 0        
    if  dt['204、我从来没有撒过谎'][i] == 5 :
        dt_lie_item ['204、我从来没有撒过谎'][i] = 0
    elif dt['204、我从来没有撒过谎'][i] == -3:
        dt_lie_item ['204、我从来没有撒过谎'][i] = -3
    else:        
        dt_lie_item ['204、我从来没有撒过谎'][i] = 1          
    if dt['102、本题请选择最左边的选项'][i] == 1:
       dt_lie_item ['102、本题请选择最左边的选项'][i] = 1
    else:        
        dt_lie_item ['102、本题请选择最左边的选项'][i] = 0     
    if  dt['51、我从来没有遇到过让我难过的事'][i] == 5 :
        dt_lie_item ['51、我从来没有遇到过让我难过的事'][i] = 0
    else:        
        dt_lie_item ['51、我从来没有遇到过让我难过的事'][i] = 1     
    if dt['153、本题请选择中间的选项'][i] == 3:
       dt_lie_item ['153、本题请选择中间的选项'][i] = 1
    else:        
        dt_lie_item ['153、本题请选择中间的选项'][i] = 0
    
dt_lie_item = pd.concat([dt['143、您看耽美吗'],dt_lie_item],axis=1) 
 
#计算耽美爱好者测谎题得分 
dt_yaoi_lie = dt_lie_item[dt_lie_item['143、您看耽美吗']==1]
yaoi_lie = dt_yaoi_lie.iloc[:,[1,2,3,4,5]]
dt_yaoi['lie_total'] = yaoi_lie.apply(lambda x: x.sum(),axis=1)
#计算耽非美爱好者测谎题得分 
dt_non_yaoi_lie =  dt_lie_item[dt_lie_item['143、您看耽美吗']==2]
non_yaoi_lie = dt_non_yaoi_lie.iloc[:,[1,2]]
dt_non_yaoi['lie_total'] = non_yaoi_lie.apply(lambda x: x.sum(),axis=1)
#计算测谎题得分
dt_total_lie = pd.concat([dt_yaoi,dt_non_yaoi],axis=0)
#挑选能用的被试
dt_yaoi_useful = dt_total_lie.loc[(dt_total_lie['143、您看耽美吗']==1)&(dt_total_lie['lie_total'] >=4)]
dt_non_yaoi_useful = dt_total_lie.loc[(dt_total_lie['143、您看耽美吗']==2)&(dt_total_lie['lie_total']==2)]

#不能用的被试
#测谎题未通过
dt_yaoi_nonuseful = dt_total_lie.loc[(dt_total_lie['143、您看耽美吗']==1)&(dt_total_lie['lie_total'] < 4)]
dt_non_yaoi_nonuseful = dt_total_lie.loc[(dt_total_lie['143、您看耽美吗']==2)&(dt_total_lie['lie_total']<2)]
nonuseful = pd.concat([dt_yaoi_nonuseful,dt_non_yaoi_nonuseful],axis = 0)
nonuseful.to_excel(str(date)+'_测谎题未通过被试.xlsx') 
dt_yaoi_useful.to_excel(str(date)+'_耽美爱好者被试费名单_10.xlsx') 
dt_non_yaoi_useful.to_excel(str(date)+'_非耽美爱好者被试费名单_5.xlsx')                          
dt_total_lie = pd.concat([dt_yaoi_useful,dt_non_yaoi_useful],axis=0)
dt_total_lie.index = range(dt_total_lie.shape[0])#所有通过测谎题的被试顺序重新排列
dt_total_lie.to_excel(str(date)+'_dt_total_lie.xlsx')

#time 判断正态性
time = dt_total_lie.iloc[:,2]
dt_time = pd.concat([pd.DataFrame( [ time.tolist()[i].replace('秒','') for i in range(dt_total_lie.shape[0]) ],columns = {'time'}),dt_total_lie],axis = 1 ) #删除秒
dt_time = pd.concat([pd.DataFrame( [np.log( int(dt_time['time'][i]) ) for i in range(dt_total_lie.shape[0])], columns = {'time_ln'}),dt_time ] ,axis = 1) #ln(time)
time_kurt_yaoi = dt_time.loc[dt_time['143、您看耽美吗']==1]['time_ln'].kurt() # 峰度：陡峭程度
time_kurt_yaoi_non = dt_time.loc[dt_time['143、您看耽美吗']==2]['time_ln'].kurt()
time_skew_yaoi = dt_time.loc[dt_time['143、您看耽美吗']==1]['time_ln'].skew() # 峰度：对称程度；越接近越是正太
time_skew_yaoi_non = dt_time.loc[dt_time['143、您看耽美吗']==2]['time_ln'].skew()
# 求Z分数
time_ln_z_yaoi = pd.DataFrame( [(dt_time.loc[dt_time['143、您看耽美吗']==1]['time_ln'][i] - 
                               dt_time.loc[dt_time['143、您看耽美吗']==1]['time_ln'].mean()) 
                              / dt_time.loc[dt_time['143、您看耽美吗']==1]['time_ln'].std() 
                              for i in range(dt_time.loc[dt_time['143、您看耽美吗']==1].shape[0]) ] )
time_ln_z_yaoi_t = dt_time.loc[dt_time['143、您看耽美吗']==2]
time_ln_z_yaoi_t.index = range(dt_time.loc[dt_time['143、您看耽美吗']==2].shape[0])
time_ln_z_yaoi_non = pd.DataFrame( [ ( time_ln_z_yaoi_t['time_ln'][i] -                                      
                               time_ln_z_yaoi_t['time_ln'].mean() ) 
                              / time_ln_z_yaoi_t['time_ln'].std() 
                              for i in range(time_ln_z_yaoi_t.shape[0]) ] )

time_ln_z_total = pd.concat( [time_ln_z_yaoi,time_ln_z_yaoi_non] , axis = 0 )
time_ln_z_total.columns = ['time_ln_z']
time_ln_z_total.index = range(dt_time.shape[0])
pd.concat([time_ln_z_total,dt_time],axis =1 )
dt_time.to_excel(str(date)+'dt_total_lie_time_ln_z.xlsx') 


#分量表以及删除测谎题,反向记分
dt_rk = dt_total_lie.iloc[:,7:35]#人口学变量
dt_rk.columns =  ['R'+str(i) for i in range(1,29)]
dt_rk2 = dt_total_lie.iloc[:,150:154]#群体歧视+区分耽美 
dt_rk2.columns = ['R'+str(i) for i in range(29,33)]
dt_rkd = dt_total_lie.iloc[:,154:167]#耽美爱好者和耽美相关的人口学变量 
del dt_rkd['153、本题请选择中间的选项']
dt_rkd.columns = ['RD'+str(i) for i in range(1,13)]



dt_distinguish = dt_total_lie.iloc[:,35:41]#6题
dt_distinguish = dt_distinguish.replace(2,0)# 2否
del dt_distinguish['27、您认为自己会喜欢，以描述男性与男性之间的爱情或性为主题的作品吗（小说、动漫、同人文、影视剧等）']
dt_distinguish.columns = ['D'+str(i) for i in range(1,6)]
dt_distinguish['D_total'] = dt_distinguish.apply(lambda x: x.sum(),axis=1)
distinguish = pd.concat([dt_total_lie['143、您看耽美吗'],dt_distinguish],axis=1)
R = pd.concat([dt_distinguish['D_total'],dt_rk,dt_rk2,time_ln_z_total],axis=1)
R.to_excel(str(date)+'_1_demography_all_item.xlsx') 

dt_rkd = pd.concat([R,dt_rkd,time_ln_z_total],axis=1)
dt_rkd.loc[(dt_rkd['R32']==1)&(dt_rkd['D_total'] > 2)].to_excel(str(date)+'_2_demography_bl_item.xlsx')
pd.concat([time_ln_z_total,distinguish],axis = 1).to_excel(str(date)+'_3_distinguish_new2.xlsx')

#辨别题异常
dt_abnormal = pd.concat([dt_distinguish['D_total'],dt_total_lie,time_ln_z_total],axis = 1 )
dt_abnormal_nonyaoi = dt_abnormal.loc[(dt_abnormal['143、您看耽美吗']==2)&(dt_abnormal['D_total'] > 2)]
dt_abnormal_yaoi = dt_abnormal.loc[(dt_abnormal['143、您看耽美吗']==1)&(dt_abnormal['D_total'] <= 2)]
dt_abnormal_nonyaoi.to_excel(str(date)+'_9_dt_abnormal_nonyaoi.xlsx')
dt_abnormal_yaoi.to_excel(str(date)+'_9_dt_abnormal_yaoi.xlsx')




dt_personality = dt_total_lie.iloc[:,42:103]
del dt_personality['51、我从来没有遇到过让我难过的事']
dt_personality.columns =  ['P'+str(i) for i in range(1,61)]#大五人格量表简版
r_item = ['P11','P16','P26','P31','P36','P51',						
'P12','P17','P22','P37','P42','P47',				
'P3','P8','P23','P28','P48','P58',				
'P4','P9','P24','P29','P44','P49',				
'P5','P25','P30','P45','P50','P55']
dt_personality_useful_rs =  Reverse_score(dt_personality,r_item)
Per_item = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,dt_personality_useful_rs],axis=1) 
Per_item.to_excel(str(date)+'_4_big_five_item.xlsx')
P_EX = ['P1','P6','P11','P16','P21','P26','P31','P36','P41','P46','P51','P56']
P_AG = ['P2','P7','P12','P17','P22','P27','P32','P37','P42','P47','P52','P57']
P_CO = ['P3','P8','P13','P18','P23','P28','P33','P38','P43','P48','P53','P58']
P_NE = ['P4','P9','P14','P19','P24','P29','P34','P39','P44','P49','P54','P59']
P_OM = ['P5','P10','P15','P20','P25','P30','P35','P40','P45','P50','P55','P60']  
dt = dt_personality_useful_rs
name = ['P_EX','P_AG','P_CO','P_NE','P_OM'] 
d_name = [P_EX,P_AG,P_CO,P_NE,P_OM]  
P = dimension_grade(name,dt,d_name)
P['P_total']=P.apply(lambda x: x.sum(),axis=1)
Per = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,P],axis=1) 
Per.loc[(Per['143、您看耽美吗']==1)&(Per['D_total'] > 2)]
Per.to_excel(str(date)+'_4_big_five.xlsx')


#12个分维度
PD_Sociability=['P1','P16','P31','P46']				
PD_Assetiveness=['P6','P21','P36','P51']					
PD_EnegyLevel=['P11','P26','P41','P56']					
PD_Compassion=['P2','P17','P32','P47']					
PD_espectfulness=['P7','P22','P37','P52']				
PD_Tust=['P12','P27','P42','P57']					
PD_Oganization=['P3','P18','P33','P48']					
PD_Poductiveness=['P8','P23','P38','P53']					
PD_esponsibility=['P13','P28','P43','P58']					
PD_Anxiety=['P4','P19','P34','P49']					
PD_Depession=['P9','P24','P39','P54']					
PD_EmotionalVolatility=['P14','P29','P44','P59']					
PD_IntellectualCuiosity=['P10','P25','P40','P55']					
PD_AestheticSensitivity=['P5','P20','P35','P50']					
PD_CeativeImagination=['P15','P30','P45','P60']					
dt = dt_personality_useful_rs
name = ['PD_Sociability','PD_Assetiveness','PD_EnegyLevel','PD_Compassion','PD_espectfulness','PD_Tust',
        'PD_Oganization','PD_Poductiveness','PD_esponsibility','PD_Anxiety','PD_Depession',
          'PD_EmotionalVolatility','PD_IntellectualCuiosity','PD_AestheticSensitivity','PD_CeativeImagination']   
d_name = [PD_Sociability,PD_Assetiveness,PD_EnegyLevel,PD_Compassion,PD_espectfulness,PD_Tust,
          PD_Oganization,PD_Poductiveness,PD_esponsibility,PD_Anxiety,PD_Depession,
          PD_EmotionalVolatility,PD_IntellectualCuiosity,PD_AestheticSensitivity,PD_CeativeImagination]  
PD = dimension_grade(name,dt,d_name)
PD['PD_total']=PD.apply(lambda x: x.sum(),axis=1)
PDr = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,PD],axis=1) 
PDr.to_excel(str(date)+'_4_big_five_divided.xlsx')


  
dt_at = dt_total_lie.iloc[:,103:124] 
del dt_at['102、本题请选择最左边的选项']
dt_at.columns = ['AT'+str(i) for i in range(1,21)]#同性恋态度量表
r_item = ['AT1','AT3','AT5','AT6','AT7','AT8','AT9','AT11','AT14','AT15','AT16','AT17','AT19','AT20']
dt_at_useful_rs =  Reverse_score(dt_at,r_item)
AT_item = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,dt_at_useful_rs],axis=1) 
AT_item.to_excel(str(date)+'_5_attitude_homosexual_item.xlsx')
AT_L = dt_at_useful_rs.iloc[:,0:10].columns.values.tolist() 
AT_G = dt_at_useful_rs.iloc[:,10:20].columns.values.tolist() 
name = ['AT_L','AT_G']   
d_dt = dt_at_useful_rs
d_name = [AT_L,AT_G]   
AT = dimension_grade(name,d_dt,d_name)
AT['AT_total']=AT.apply(lambda x: x.sum(),axis=1)
ATT = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,AT],axis=1) 
ATT.to_excel(str(date)+'_5_attitude_homosexual.xlsx')




dt_eci = dt_total_lie.iloc[:,124:150] 
dt_eci.columns = ['E'+str(i) for i in range(1,27)]#情绪创造性量表
r_item = ['E4']
dt_eci_useful_rs =  Reverse_score(dt_eci,r_item)
eci_item = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,dt_eci_useful_rs],axis=1) 
eci_item.to_excel(str(date)+'_6_emotion_creativity_item.xlsx')
ECI_P = dt_eci_useful_rs.iloc[:,0:7].columns.values.tolist() 
ECI_N = dt_eci_useful_rs.iloc[:,7:20].columns.values.tolist() 
ECI_EA = dt_eci_useful_rs.iloc[:,20:26].columns.values.tolist() 
name = ['ECI_P','ECI_N','ECI_EA']   
d_dt = dt_eci_useful_rs
d_name = [ECI_P,ECI_N,ECI_EA]   
ECI = dimension_grade(name,d_dt,d_name)
ECI['ECI_total']=ECI.apply(lambda x: x.sum(),axis=1)
eci = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,ECI],axis=1) 
eci.to_excel(str(date)+'_6_emotion_creativity.xlsx')




dt_motive = dt_total_lie.iloc[:,167:222] 
del dt_motive['204、我从来没有撒过谎']
dt_motive.columns = ['M'+str(i) for i in range(1,55)]#耽美动机量表
motive_item = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,dt_motive],axis=1) 
motive_item.to_excel(str(date)+'_7_motive_yaoi_item.xlsx')
motive_item.to_csv(str(date)+'7_motive_yaoi_item.csv')
M_PL=['M1','M2','M3','M4','M5','M43']
M_IA_ME=['M6','M7','M8','M9','M10']
M_DS=['M11','M12','M13']
M_FO=['M14','M15','M16','M17']
M_ER=['M18','M19','M20','M21','M36','M37','M38','M39']
M_AA_PE=['M22','M23','M24','M25','M26','M27','M28','M40']
M_AS=['M29','M30','M31','M32','M33','M34','M35','M41','M42','M44']
M_OF=['M52','M53','M54']
M_TP=['M45','M46','M47','M48','M49','M50','M51']
name = ['M_PL','M_IA_ME','M_DS','M_FO','M_ER','M_AA_PE','M_AS','M_OF','M_TP']   
d_dt = dt_motive
d_name = [M_PL,M_IA_ME,M_DS,M_FO,M_ER,M_AA_PE,M_AS,M_OF,M_TP]   
M = dimension_grade(name,d_dt,d_name)
M['M_total']=M.apply(lambda x: x.sum(),axis=1)
m = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,M],axis=1) 
m[(m['143、您看耽美吗']==1)].to_excel(str(date)+'_7_motive_yaoi.xlsx')
m[(m['143、您看耽美吗']==1)].to_csv(str(date)+'_7_motive_yaoi.csv')


dt_addict = dt_total_lie.iloc[:,222:267] 
del dt_addict['250、本题请选择最右边的选项']
del dt_addict['233、我认为自己是一个耽美爱好者']
dt_addict.columns = ['A'+str(i) for i in range(1,44)]#耽美成瘾量表
addict_item = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,dt_addict],axis=1) 
addict_item.to_excel(str(date)+'_8_addict_yaoi_item.xlsx')
addict_item.to_csv(str(date)+'_8_addict_yaoi_item.csv')
A_RS=['A1','A3','A19','A40','A41','A42','A43']
A_PF=['A2','A5','A6','A7','A8','A9']
A_WD=['A10','A11','A13','A20','A21']
A_TM=['A16','A17','A18','A22','A23']
A_FS=['A33','A34','A38']
A_CS=['A4','A12','A14','A15','A24','A25','A26','A27','A28','A29','A30','A31','A32','A35','A36','A37','A39']
name = ['A_RS','A_PF','A_WD','A_TM','A_FS','A_CS']   
d_dt = dt_addict
d_name = [A_RS,A_PF,A_WD,A_TM,A_FS,A_CS]   
A = dimension_grade(name,d_dt,d_name)
A['A_total']=A.apply(lambda x: x.sum(),axis=1)
a = pd.concat([dt_total_lie['143、您看耽美吗'],distinguish['D_total'],time_ln_z_total,A],axis=1) 
a[(a['143、您看耽美吗']==1)].to_excel(str(date)+'_8_addict_yaoi.xlsx')
a[(a['143、您看耽美吗']==1)].to_csv(str(date)+'_8_addict_yaoi.csv')
lb_all_dt = pd.concat([time_ln_z_total,dt_total_lie.iloc[:,0:2],R,dt_distinguish,dt_rkd,P,PD,AT,ECI,M,A],axis=1) #序号，提交时间
lb_all_dt.to_excel(str(date)+'_all_divided.xlsx')




















        
