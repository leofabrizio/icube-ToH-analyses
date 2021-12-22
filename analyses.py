# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:14:58 2020

@author: FLeo
"""

from statistics import mean
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import chisquare


#load dataset                                  
# path = 'C:\\Users\\FLeo\\Downloads\\icube-master\\icube_device\\saved_data\\exp1_joint_action\\'
file_name = 'data.csv'
df = pd.read_csv(file_name)
#df.drop(df.columns[0], axis=1, inplace=True)

# df_sub = df[df['cond'] == 'alone'] #seleziono subset x cond ALONE
#df_sub = df[df['accuracy']==1]
# df_sub = df[df['MRT']<=13]
# df_sub = df[df['code'] == 11] 
# df_sub = df[df['cluster'] == 0] 

# df_sub.reset_index(drop=True,inplace=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#plot ACCURACY
pg.normality(df['accuracy'])
data = df['accuracy'].values
plt.bar('alone',np.mean(data),yerr=stats.sem(data))
plt.title('ACCURACY')
# plt.xlabel('Face type')
plt.ylabel('Accuracy (prop)')
plt.xticks([])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#plot EXPLOR DURATION
pg.normality(df['duration_trial'])
data = df['duration_trial'].values

w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
# plt.bar('alone',np.mean(data),yerr=stats.sem(data))
plt.boxplot(data,showmeans=True)
plt.title('EXPLORATION DURATION')
# plt.xlabel('Face type')
plt.ylabel('Exploration Duration (s)')
plt.xticks([])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#plot FILT N° TOCCHI TRIAL
pg.normality(df['filt_n_tocchi_trial'])
data = df['filt_n_tocchi_trial'].values

w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
plt.boxplot(data,showmeans=True)
plt.title('FILTERED N° TOUCHES PER TRIAL')
# plt.xlabel('Face type')
plt.ylabel('N° Touches')
plt.xticks([])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#plot FILT TOUCH FREQ TRIAL
pg.normality(df['filt_touch_density_trial'])
data = df['filt_touch_density_trial'].values

w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
# plt.bar('alone',np.mean(data),yerr=stats.sem(data))
plt.boxplot(data,showmeans=True)
plt.title('FILTERED TOUCH FREQUENCY')
# plt.xlabel('Face type')
plt.ylabel('N° Touches / s')
plt.xticks([])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#plot amount ROTATION
pg.normality(df['rot_abs_total'])
data = df['rot_abs_total'].values

w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
# plt.bar('alone',np.mean(data),yerr=stats.sem(data))
plt.boxplot(data,showmeans=True)

plt.title('MEAN MAXIMUM ROTATION')
# plt.xlabel('Face type')
plt.ylabel('Max Rotation (°)')
plt.xticks([])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#plot FILT ROTATION VELOCITY
pg.normality(df['filtered_mean_rot_velocity'])
data = df['filtered_mean_rot_velocity'].values

w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
# plt.bar('alone',np.mean(data),yerr=stats.sem(data))
plt.boxplot(data,showmeans=True)
plt.title('MEAN ROTATION VELOCITY')
# plt.xlabel('Face type')
plt.ylabel('Degrees / s')
plt.xticks([])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation betwen n° pins and exploration duration
lista1 = [df['pin_facet_1'],df['pin_facet_2'],df['pin_facet_3'],df['pin_facet_4'],
             df['pin_facet_5'],df['pin_facet_6']]    
lista_pin = [item for sublist in lista1 for item in sublist]        

lista2 = [df['filt_dur_facet_1'],df['filt_dur_facet_2'],df['filt_dur_facet_3'],df['filt_dur_facet_4'],
             df['filt_dur_facet_5'],df['filt_dur_facet_6']]
lista_dur = [item for sublist in lista2 for item in sublist]     
# corr = stats.pearsonr(lista_dur,lista_pin)  

pg.normality(lista_dur)  #non normale

corr = pg.corr(lista_dur,lista_pin,method='spearman').round(3)

#plot
ax = sns.regplot(x=lista_dur, y=lista_pin, color="b") 
ax.set(xlabel='Duration (s)', ylabel='Number of pins', title='Correlation')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube accuracy and explor duration
corr=pg.corr(df['duration_trial'], df['accuracy'], method='spearman').round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['duration_trial'], y=df['accuracy'], color="b") 
ax.set(xlabel='Exploration Duration (s)', ylabel='Accuracy', title='Correlation between Exploration Duration & Accuracy')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube accuracy and n° touches
corr=pg.corr(df['filt_n_tocchi_trial'], df['accuracy'], method='spearman').round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['filt_n_tocchi_trial'], y=df['accuracy'], color="b") 
ax.set(xlabel='N° Touches', ylabel='Accuracy', title='Correlation between N° Touches & Accuracy')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube accuracy and touch frequency
corr=pg.corr(df['filt_touch_density_trial'], df['accuracy'], method='spearman').round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['filt_touch_density_trial'], y=df['accuracy'], color="b") 
ax.set(xlabel='Touch Frequency (degrees / s)', ylabel='Accuracy', title='Correlation between Touch Frequency & Accuracy')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube accuracy and amount rotation
corr=pg.corr(df['rot_abs_total'], df['accuracy'], method='spearman').round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['rot_abs_total'], y=df['accuracy'], color="b") 
ax.set(xlabel='Amount of Rotation (°)', ylabel='Accuracy', title='Correlation between Amount of Rotation & Accuracy')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube accuracy and rot velocity
corr=pg.corr(df['filtered_mean_rot_velocity'], df['accuracy'], method='spearman').round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['filtered_mean_rot_velocity'], y=df['accuracy'], color="b") 
ax.set(xlabel='Degrees / s', ylabel='Accuracy', title='Correlation between Rotation Velocity & Accuracy')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube exploration duration and num touches 
corr=pg.corr(df['filt_n_tocchi_trial'], df['duration_trial']).round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['filt_n_tocchi_trial'], y=df['duration_trial'], color="b") 
ax.set(xlabel='N° touches', ylabel='Duration (s)', title='Number of Touches & Exploration Duration')
ax.set(ylim=(15, 90))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation betwen iCube exploration duration and touch frequency
corr=pg.corr(df['filt_touch_density_trial'], df['duration_trial']).round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['filt_touch_density_trial'], y=df['duration_trial'], color="b") 
ax.set(xlabel='Touch Frequency (N° touches/s)', ylabel='Duration (s)', title='Touch Frequency & Exploration Duration')
ax.set(ylim=(15, 90))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation betwen iCube exploration duration e amount rotation 
corr=pg.corr(df['rot_abs_total'], df['duration_trial']).round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['rot_abs_total'], y=df['duration_trial'], color="b") 
ax.set(xlabel='Amount of Rotation (°)', ylabel='Duration (s)', title='Amount of Rotation & Exploration Duration')
ax.set(ylim=(15, 90))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube exploration duration e rotation velocity
corr=pg.corr(df['filtered_mean_rot_velocity'], df['duration_trial']).round(4)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['filtered_mean_rot_velocity'], y=df['duration_trial'], color="b") 
ax.set(xlabel='Rotation velocity (degrees / s)', ylabel='Duration (s)', title='Rotation Velocity & Exploration Duration')
ax.set(ylim=(15, 90))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Check difference in N°TOUCHES, TOUCH FREQ and DURATION by faces corresponding to own task vs the other task
d_task = df.loc[:,'task']
d_cond = df.loc[:,'cond_facet_1':'cond_facet_6']
d_tocchi = df.loc[:,'filt_n_tocchi_facet_1':'filt_n_tocchi_facet_6']
d_freq = df.loc[:,'filt_touch_density_facet_1':'filt_touch_density_facet_6']
d_dur = df.loc[:,'filt_dur_facet_1':'filt_dur_facet_6']

tocchi_task = []
tocchi_notask = []
dur_task = []
dur_notask = []
freq_task = []
freq_notask = []

for row in range(d_task.shape[0]):
    for column in range(d_cond.shape[1]):
        if d_cond.iloc[row,column] == d_task.iloc[row]:
            tocchi_task.append(d_tocchi.iloc[row,column])
            dur_task.append(d_dur.iloc[row,column])
            freq_task.append(d_freq.iloc[row,column])
        elif d_cond.iloc[row,column] != d_task.iloc[row]:   
            tocchi_notask.append(d_tocchi.iloc[row,column])
            dur_notask.append(d_dur.iloc[row,column])
            freq_notask.append(d_freq.iloc[row,column])

print('Il numero medio dei tocchi task è: ', mean(tocchi_task),'+-',stats.sem(tocchi_task),'SEM')
print('Il numero medio dei tocchi no task è: ', mean(tocchi_notask),'+-',stats.sem(tocchi_notask),'SEM')
print('la touch frequency media dei tocchi task è: ', mean(freq_task),'+-',stats.sem(freq_task),'SEM')
print('la touch frequency media dei tocchi no task è: ', mean(freq_notask),'+-',stats.sem(freq_notask),'SEM')
print('la durata dei tocchi task è: ', mean(dur_task),'+-',stats.sem(dur_task),'SEM')
print('la durata dei tocchi no task è: ', mean(dur_notask),'+-',stats.sem(dur_notask),'SEM')


#check normality
pg.normality(tocchi_task)  #non normale
pg.normality(tocchi_notask)  #non normale
pg.normality(dur_task)  #non
pg.normality(dur_notask)  #non
pg.normality(freq_task)  #non
pg.normality(freq_notask)  #non

# stats.ttest_rel(tocchi_task,tocchi_notask)
stats.ttest_rel(dur_task,dur_notask)
#stats.ttest_rel(freq_task,freq_notask)

stats.wilcoxon(tocchi_task,tocchi_notask)
stats.wilcoxon(dur_task,dur_notask)
stats.wilcoxon(freq_task,freq_notask)

#plot n° touches by kind of face
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
plt.bar(['task','no task'],[mean(tocchi_task),mean(tocchi_notask)],yerr=(stats.sem(tocchi_task),stats.sem(tocchi_notask)),color=['r','g'],capsize=7)
plt.title('Filtered N° Touches per face type')
plt.xlabel('Face type')
plt.ylabel('N° Touches')

plt.show()

#plot exploration duration by kind of face
plt.figure(figsize=(w,h),dpi=d)
plt.bar(['task','no task'],[mean(dur_task),mean(dur_notask)],yerr=(stats.sem(dur_task),stats.sem(dur_notask)),color=['r','g'],capsize=7)
plt.title('Filt Exploration duration per face type')
plt.xlabel('Face type')
plt.ylabel('Duration (s)')

plt.show()

#plot touch frequency by kind of face
plt.figure(figsize=(w,h),dpi=d)
plt.bar(['task','no task'],[mean(freq_task),mean(freq_notask)],yerr=(stats.sem(freq_task),stats.sem(freq_notask)),color=['r','g'],capsize=7)
plt.title('Touch Frequency per face type')
plt.xlabel('Face type')
plt.ylabel('N° Touches/s')

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube duration exploration and MRT score 
df_sub = df[df['accuracy']==1]
#check normality
pg.normality(df_sub['duration_trial'])  #normale
pg.normality(df_sub['MRT'])  #non normale
corr=pg.corr(df_sub['duration_trial'], df_sub['MRT'],method='spearman').round(5)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df_sub['duration_trial'], y=df_sub['MRT'], color="b") 
ax.set(xlabel='Exploration Duration (s)', ylabel='MRT', title='Exploration Duration & MRT scores')
ax.set(ylim=(5, 22.5))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between number of touches and MRT score 
df_sub = df[df['accuracy']==1]
corr=pg.corr(df_sub['filt_n_tocchi_trial'], df_sub['MRT'],method='spearman').round(3)
#plot
ax = sns.regplot(x=df_sub['filt_n_tocchi_trial'], y=df_sub['MRT'], color="b") 
ax.set(xlabel='Number of Touches', ylabel='MRT', title='Correlation between Number of Touches & MRT scores')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation betwen touch frequency and MRT score 
df_sub = df_sub[df_sub['accuracy']==1]
corr=pg.corr(df_sub['filt_touch_density_trial'], df_sub['MRT'],method='spearman').round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df_sub['filt_touch_density_trial'], y=df_sub['MRT'], color="b") 
ax.set(xlabel='Touch Frequency (n° touches/s)', ylabel='MRT', title='Touch Frequency & MRT scores')
ax.set(ylim=(5, 22.5))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube rotation velocity and MRT score 
corr=pg.corr(df['filtered_mean_rot_velocity'], df['MRT'],method='spearman').round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['filtered_mean_rot_velocity'], y=df['MRT'], color="b") 
ax.set(xlabel='Rotation Velocity (°/s)', ylabel='MRT', title='Rotation Velocity & MRT scores')
ax.set(ylim=(5, 22.5))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube accuracy and MRT score 
acc_all = df['accuracy'].values
mrt_all = df['MRT'].values
stats.pointbiserialr(acc_all, mrt_all)

group = []
for i in range(len(acc_all)):
    if acc_all[i]==1:
        group.append('correct')
    elif acc_all[i]==0: 
        group.append('wrong')

#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.boxplot(group,mrt_all,showmeans=True, order=['wrong','correct'])
ax.set(xlabel='Response', ylabel='MRT', title='Sum Accuracy & MRT scores')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube amount rotation and MRT score
corr=pg.corr(df['rot_abs_total'], df['MRT']).round(3)
#plot
ax = sns.regplot(x=df['rot_abs_total'], y=df['MRT'], color="b") 
ax.set(xlabel='Mean Max Rotation', ylabel='MRT', title='Correlation between Mean Max Rotation & MRT scores')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#check for learning effects
#select variables to check for learning effects 
df_sub1 = df.loc[:,['code','trial','accuracy','filt_n_tocchi_trial','filt_touch_density_trial',
                        'duration_trial','rot_abs_total','filtered_mean_rot_velocity']]

acc_trial2 = []
acc_trial3 = []
n_tocchi_trial2 = []
n_tocchi_trial3 = []
touch_frequency_trial2 = []
touch_frequency_trial3 = []
duration_trial2 = []
duration_trial3 = []
max_rot_trial2 = []
max_rot_trial3 = []
rot_vel_trial2 = []
rot_vel_trial3 = []

for code in np.unique(df_sub1['code']):
    df_new = df_sub1[df_sub1['code']==code]
    df_new.reset_index(drop=True,inplace=True)
    acc_trial2.append(df_new['accuracy'][1]-df_new['accuracy'][0])
    acc_trial3.append(df_new['accuracy'][2]-df_new['accuracy'][0])
    n_tocchi_trial2.append(df_new['filt_n_tocchi_trial'][1]-df_new['filt_n_tocchi_trial'][0])
    n_tocchi_trial3.append(df_new['filt_n_tocchi_trial'][2]-df_new['filt_n_tocchi_trial'][0])
    touch_frequency_trial2.append(df_new['filt_touch_density_trial'][1]-df_new['filt_touch_density_trial'][0])
    touch_frequency_trial3.append(df_new['filt_touch_density_trial'][2]-df_new['filt_touch_density_trial'][0])
    duration_trial2.append(df_new['duration_trial'][1]-df_new['duration_trial'][0])
    duration_trial3.append(df_new['duration_trial'][2]-df_new['duration_trial'][0])
    max_rot_trial2.append(df_new['rot_abs_total'][1]-df_new['rot_abs_total'][0])
    max_rot_trial3.append(df_new['rot_abs_total'][2]-df_new['rot_abs_total'][0])
    rot_vel_trial2.append(df_new['filtered_mean_rot_velocity'][1]-df_new['filtered_mean_rot_velocity'][0])
    rot_vel_trial3.append(df_new['filtered_mean_rot_velocity'][2]-df_new['filtered_mean_rot_velocity'][0])

#check normality
pg.normality(acc_trial2) #not
pg.normality(acc_trial3) #not
pg.normality(n_tocchi_trial2) #not
pg.normality(n_tocchi_trial3) #true
pg.normality(touch_frequency_trial2) #true
pg.normality(touch_frequency_trial3) #true
pg.normality(duration_trial2) #true
pg.normality(duration_trial3) #true
pg.normality(max_rot_trial2) #true
pg.normality(max_rot_trial3) #true
pg.normality(rot_vel_trial2) #true

#check difference compared to 0 (all not significant)
stats.wilcoxon(acc_trial2) 
stats.wilcoxon(acc_trial3) 
stats.wilcoxon(n_tocchi_trial2) 
stats.ttest_1samp(n_tocchi_trial3,0.0)
stats.ttest_1samp(touch_frequency_trial2,0.0)
stats.ttest_1samp(touch_frequency_trial3,0.0)
stats.ttest_1samp(duration_trial2,0.0)
stats.ttest_1samp(duration_trial3,0.0)
stats.ttest_1samp(max_rot_trial2,0.0)
stats.ttest_1samp(max_rot_trial3,0.0)
stats.ttest_1samp(rot_vel_trial2,0.0)
stats.ttest_1samp(rot_vel_trial3,0.0)

#plot accuracy
plt.bar(['trial 2','trial 3'],[np.mean(acc_trial2),np.mean(acc_trial3)],yerr=(stats.sem(acc_trial2),stats.sem(acc_trial3)),capsize=7)
plt.title('ACCURACY Learning Effect (difference compared to baseline)')
plt.ylabel('Accuracy Difference (prop)')
plt.show()

#plot n° touches
plt.bar(['trial 2','trial 3'],[np.mean(n_tocchi_trial2),np.mean(n_tocchi_trial3)],yerr=(stats.sem(n_tocchi_trial2),stats.sem(n_tocchi_trial3)),capsize=7)
plt.title('N° TOUCHES Learning Effect (difference compared to baseline)')
plt.ylabel('N° Touches Difference')
plt.show()

#plot exploration duration
plt.bar(['trial 2','trial 3'],[np.mean(duration_trial2),np.mean(duration_trial3)],yerr=(stats.sem(duration_trial2),stats.sem(duration_trial3)),capsize=7)
plt.title('EXP DURATION Learning Effect (difference compared to baseline)')
plt.ylabel('Exploration Duration Difference')
plt.show()

#plot touch frequency
plt.bar(['trial 2','trial 3'],[np.mean(touch_frequency_trial2),np.mean(touch_frequency_trial3)],yerr=(stats.sem(touch_frequency_trial2),stats.sem(touch_frequency_trial3)),capsize=7)
plt.title('TOUCH FREQUENCY Learning Effect (difference compared to baseline)')
plt.ylabel('Touch Frequency Difference')
plt.show()

#plot amount of rotation
plt.bar(['trial 2','trial 3'],[np.mean(max_rot_trial2),np.mean(max_rot_trial3)],yerr=(stats.sem(max_rot_trial2),stats.sem(max_rot_trial3)),capsize=7)
plt.title('MAX ROTATION Learning Effect (difference compared to baseline)')
plt.ylabel('Max Rotation Difference')
plt.show()

#plot rotation velocity
plt.bar(['trial 2','trial 3'],[np.mean(rot_vel_trial2),np.mean(rot_vel_trial3)],yerr=(stats.sem(rot_vel_trial2),stats.sem(rot_vel_trial3)),capsize=7)
plt.title('ROTATION VELOCITY Learning Effect (difference compared to baseline)')
plt.ylabel('Rotation Velocity Difference')
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#check whether n° returns depend on kind of face 
compito = df['task']
tipo_faccia = df.loc[:,['cond_facet_1','cond_facet_2','cond_facet_3','cond_facet_4',
                            'cond_facet_5','cond_facet_6']]

#liste verifica differenza nel n° ritorni in base al tipo di faccia (even o odd)
task_list = []
notask_list = []

#seleziono e converto array ritorni
for indice in range(len(df)):
    seq = df.loc[indice,'ritorni_0.8s']
    seq = seq.replace('[','')
    seq = seq.replace(']','')
    seq = np.fromstring(seq, dtype=int, sep=',')
    
    for indice1,elemento in enumerate(seq):
        # print(indice1,elemento)
        if elemento != 0:
            if compito[indice] == tipo_faccia.iloc[indice,indice1]:
                task_list.append(elemento)
            elif compito[indice] != tipo_faccia.iloc[indice,indice1]:
                notask_list.append(elemento)
                
somma_task = sum(task_list)
somma_notask = sum(notask_list)

chisquare([somma_task,somma_notask]) #non ci sono differenze in base al tipo faccia [even vs odd]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#check whether n° returns depend on n° pins

options1 = [1,8,3,4,15,13,11,6,20,18]  #group conf1,2,3
options2 = [7,2,9,10,16,14,5,12,19,17]    #group conf10,11,12

# selecting rows based on condition 
df_sub1 = df[df['code'].isin(options1)] 
df_sub1.reset_index(drop=True,inplace=True)

df_sub2 = df[df['code'].isin(options2)] 
df_sub2.reset_index(drop=True,inplace=True)

#initialize arrays with n° returns by face
array_ritorni1 = np.zeros(6, dtype=int)
array_ritorni2 = np.zeros(6, dtype=int)

#slect and convert return arrays
for indice in range(len(df_sub1)):
    seq = df_sub1.loc[indice,'ritorni_0.8s']
    seq=seq.replace('[','')
    seq=seq.replace(']','')
    seq = np.fromstring(seq, dtype=int, sep=',')
    array_ritorni1 = array_ritorni1 + seq

for indice in range(len(df_sub2)):
    seq = df_sub2.loc[indice,'ritorni_0.8s']
    seq=seq.replace('[','')
    seq=seq.replace(']','')
    seq = np.fromstring(seq, dtype=int, sep=',')
    array_ritorni2 = array_ritorni2 + seq

ritorni_totali1 = np.sum(array_ritorni1)
ritorni_totali2 = np.sum(array_ritorni2)

occorrenza_facce1 = [20,30,20,40,50,20] #how many faces with 1, 2... 6 pins have been presented in all trials with conf1,2,3
occorrenza_facce2 = [20,20,30,70,40,0] #how many faces with 1, 2... 6 pins have been presented in all trials with conf10,11,12

facce_totali1 = sum(occorrenza_facce1)
facce_totali2 = sum(occorrenza_facce2)

prob_faccia_1pin1 = (occorrenza_facce1[0]/facce_totali1)*100
prob_faccia_2pin1 = (occorrenza_facce1[1]/facce_totali1)*100
prob_faccia_3pin1 = (occorrenza_facce1[2]/facce_totali1)*100
prob_faccia_4pin1 = (occorrenza_facce1[3]/facce_totali1)*100
prob_faccia_5pin1 = (occorrenza_facce1[4]/facce_totali1)*100
prob_faccia_6pin1 = (occorrenza_facce1[5]/facce_totali1)*100

prob_faccia_1pin2 = (occorrenza_facce2[0]/facce_totali2)*100
prob_faccia_2pin2 = (occorrenza_facce2[1]/facce_totali2)*100
prob_faccia_3pin2 = (occorrenza_facce2[2]/facce_totali2)*100
prob_faccia_4pin2 = (occorrenza_facce2[3]/facce_totali2)*100
prob_faccia_5pin2 = (occorrenza_facce2[4]/facce_totali2)*100
prob_faccia_6pin2 = (occorrenza_facce2[5]/facce_totali2)*100

# freq_osservata_faccia = [somma_1pin, somma_2pin, somma_3pin, somma_4pin, somma_5pin, somma_6pin]
# freq_osservata_faccia_normalizzata = [9,9,14,21,18,6] #normalizzata per equiprobabilità presentazione facce
# prob_ritorno_faccia_1pin = (freq_osservata_faccia[0]/sum(freq_osservata_faccia))*100
# prob_ritorno_faccia_2pin = (freq_osservata_faccia[1]/sum(freq_osservata_faccia))*100
# prob_ritorno_faccia_3pin = (freq_osservata_faccia[2]/sum(freq_osservata_faccia))*100
# prob_ritorno_faccia_4pin = (freq_osservata_faccia[3]/sum(freq_osservata_faccia))*100
# prob_ritorno_faccia_5pin = (freq_osservata_faccia[4]/sum(freq_osservata_faccia))*100
# prob_ritorno_faccia_6pin = (freq_osservata_faccia[5]/sum(freq_osservata_faccia))*100

# freq_attesa_faccia = ritorni_totali/6 #caso di equiprobabilità
# attesa_facce = [freq_attesa_faccia,freq_attesa_faccia,freq_attesa_faccia,freq_attesa_faccia,freq_attesa_faccia,freq_attesa_faccia]
attesa_facce_pesata1 = [round((ritorni_totali1/100)*prob_faccia_1pin1),round((ritorni_totali1/100)*prob_faccia_2pin1),
                       round((ritorni_totali1/100)*prob_faccia_3pin1),round((ritorni_totali1/100)*prob_faccia_4pin1),
                       round((ritorni_totali1/100)*prob_faccia_5pin1),round((ritorni_totali1/100)*prob_faccia_6pin1)]

attesa_facce_pesata2 = [round((ritorni_totali2/100)*prob_faccia_1pin2),round((ritorni_totali2/100)*prob_faccia_2pin2),
                       round((ritorni_totali2/100)*prob_faccia_3pin2),round((ritorni_totali2/100)*prob_faccia_4pin2),
                       round((ritorni_totali2/100)*prob_faccia_5pin2),round((ritorni_totali2/100)*prob_faccia_6pin2)]

array_ritorni_tot = array_ritorni1 + array_ritorni2
attesa_facce_pesata1 = np.array(attesa_facce_pesata1)
attesa_facce_pesata2 = np.array(attesa_facce_pesata2)
attesa_facce_pesata_tot = attesa_facce_pesata1 + attesa_facce_pesata2
chisquare(array_ritorni_tot, f_exp=attesa_facce_pesata_tot)  

#plot attended and observed distributions
labels = ['1 pin', '2 pin', '3 pin', '4 pin', '5 pin', '6 pin']
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(dpi=300)
rects1 = ax.bar(x - width/2, array_ritorni_tot, width, label='Observed Frequency')
rects2 = ax.bar(x + width/2, attesa_facce_pesata_tot, width, label='Expected Frequency')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Returns')
ax.set_ylim((0,60))
ax.set_title('Observed and expected number of returns by n° of pins per face')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#check whether n° returns correlates with accuracy and exploration duration

n_ritorni = []
#seleziono, converto e sommo arrays ritorni
for indice in range(len(df)):
    seq = df.loc[indice,'ritorni_0.8s']
    seq=seq.replace('[','')
    seq=seq.replace(']','')
    seq = np.fromstring(seq, dtype=int, sep=',')
    somma = np.sum(seq)
    n_ritorni.append(somma)
    # print(somma)
    
# correlation between n° returns and accuracy
stats.pointbiserialr(n_ritorni, df['accuracy'])

group = []
for i in range(len(df['accuracy'])):
    if df.loc[i,'accuracy']==1:
        group.append('correct')
    elif df.loc[i,'accuracy']==0: 
        group.append('wrong')

#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.boxplot(group,n_ritorni,showmeans=True, order=['wrong','correct'])
ax.set(xlabel='Response', ylabel='N° returns', title='Accuracy & Number of Returns')

# correlation between n° returns and exploration duration
pg.normality(n_ritorni)

corr=pg.corr(n_ritorni, df['duration_trial'],method='spearman').round(6)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=n_ritorni, y=df_sub['duration_trial'], color="b") 
ax.set(xlabel='N° returns', ylabel='Exploration Duration (s)', title='Exploration Duration & Number of Returns')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube mean delta timepoints and MRT score 
# df_sub = df_sub[df_sub['accuracy']==1]
#check normality
pg.normality(df['mean_delta_timepoints'])  #non normale
pg.normality(df['MRT'])  #non normale
corr=pg.corr(df['mean_delta_timepoints'], df['MRT'],method='spearman').round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=['mean_delta_timepoints'], y=['MRT'], color="b") 
ax.set(xlabel='Mean Change Face Rate (s)', ylabel='MRT', title='Mean Change Face Rate & MRT scores')
ax.set(ylim=(2.5, 22.5))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube std delta timepoints and MRT score 
# df_sub = df_sub[df_sub['accuracy']==1]
#check normality
pg.normality(df['std_delta_timepoints'])  #non normale
pg.normality(df['MRT'])  #non normale
corr=pg.corr(df['std_delta_timepoints'], df['MRT'],method='spearman').round(5)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['std_delta_timepoints'], y=df['MRT'], color="b") 
ax.set(xlabel='Std Change Face Rate (s)', ylabel='MRT', title='Variability in Change Face Rate & MRT scores')
ax.set(ylim=(2.5, 22.5))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube mean delta timepoints and accuracy
# df_sub = df_sub[df_sub['accuracy']==1]
#check normality
corr=pg.corr(df_sub['mean_delta_timepoints'], df_sub['accuracy'],method='spearman').round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df_sub['mean_delta_timepoints'], y=df_sub['accuracy'], color="b") 
ax.set(xlabel='Mean Change Face Rate (s)', ylabel='Accuracy', title='Correlation between Mean Change Face Frequency & Accuracy in the ALONE condition')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between iCube std delta timepoints and accuracy
# df_sub = df_sub[df_sub['accuracy']==1]
#check normality
corr=pg.corr(df_sub['std_delta_timepoints'], df_sub['accuracy'],method='spearman').round(3)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df_sub['std_delta_timepoints'], y=df_sub['accuracy'], color="b") 
ax.set(xlabel='Mean Change Face Rate (s)', ylabel='Accuracy', title='Correlation between Mean Change Face Frequency & Accuracy in the ALONE condition')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#check whether n° returns differs in different clusters (repeat the code for each cluster)
df_cluster0 = df[df['cluster']==0]
df_cluster1 = df[df['cluster']==1]
df_cluster0.reset_index(drop=True,inplace=True)
df_cluster1.reset_index(drop=True,inplace=True)

#lists n° returns by cluster
ritorni_cluster0 = []
ritorni_cluster1 = []

#seldvt and convert returns arrays
for indice in range(len(df_cluster0)):
    # print(indice)
    seq = df_cluster0.loc[indice,'ritorni_0.8s']
    seq=seq.replace('[','')
    seq=seq.replace(']','')
    seq = np.fromstring(seq, dtype=int, sep=',')
    print(seq)
    
    for indice1,elemento in enumerate(seq):
        # print(indice1,elemento)
        if elemento != 0:
            ritorni_cluster0.append(elemento)
            # if compito[indice] == tipo_faccia.iloc[indice,indice1]:
            #     task_list.append(elemento)
            # elif compito[indice] != tipo_faccia.iloc[indice,indice1]:
            #     notask_list.append(elemento)
                
somma_cluster0 = sum(ritorni_cluster0)
somma_cluster1 = sum(ritorni_cluster1)

chisquare([somma_cluster0,somma_cluster1]) 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#GRAF FIG12: correlation between exploration duration and n° diff transitions
corr=pg.corr(df['duration_trial'], df['diff_trans'],method='pearson').round(7)
#plot
w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.regplot(x=df['duration_trial'], y=df['diff_trans'], color="b") 
ax.set(xlabel='Duration (s)', ylabel='Number of Different Transitions', title='Number of Different Transitions & Exploration Duration')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# n° returns by subject
df_cluster = df[df['code']==20]
df_cluster.reset_index(drop=True,inplace=True)

#liste verifica differenza nel n° ritorni in base al cluster
ritorni_cluster = []

#seleziono e converto array ritorni
for indice in range(len(df_cluster)):
    # print(indice)
    seq = df_cluster.loc[indice,'ritorni_0.8s']
    seq = seq.replace('[','')
    seq = seq.replace(']','')
    seq = np.fromstring(seq, dtype=int, sep=',')
    print(seq)
    
    for indice1,elemento in enumerate(seq):
        # print(indice1,elemento)
        if elemento != 0:
            ritorni_cluster.append(elemento)
            # if compito[indice] == tipo_faccia.iloc[indice,indice1]:
            #     task_list.append(elemento)
            # elif compito[indice] != tipo_faccia.iloc[indice,indice1]:
            #     notask_list.append(elemento)
                
somma_cluster = sum(ritorni_cluster)
print(somma_cluster)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#check whether subjects who did the experiment before the additiona ltask differ
#than subjects who did the experiment after

df_prima = df[df['order']==1]
df_dopo = df[df['order']==2]

#accuracy
mean_acc_df_prima = df_prima.groupby('code')['accuracy'].mean()
mean_acc_df_dopo = df_dopo.groupby('code')['accuracy'].mean()
pg.normality(mean_acc_df_prima)
pg.normality(mean_acc_df_dopo)
stats.mannwhitneyu(mean_acc_df_prima,mean_acc_df_dopo) #p = 0.073

#duration
mean_duration_df_prima = df_prima.groupby('code')['duration_trial'].mean()
mean_duration_df_dopo = df_dopo.groupby('code')['duration_trial'].mean()
pg.normality(mean_duration_df_prima)
pg.normality(mean_duration_df_dopo)
stats.levene(mean_duration_df_prima,mean_duration_df_dopo)
stats.ttest_ind(mean_duration_df_prima,mean_duration_df_dopo, equal_var=True) #p = 0.0507

#n_touches
mean_touches_df_prima = df_prima.groupby('code')['filt_n_tocchi_trial'].mean()
mean_touches_df_dopo = df_dopo.groupby('code')['filt_n_tocchi_trial'].mean()
pg.normality(mean_touches_df_prima)
pg.normality(mean_touches_df_dopo)
stats.levene(mean_touches_df_prima,mean_touches_df_dopo)
stats.ttest_ind(mean_touches_df_prima,mean_touches_df_dopo, equal_var=True) #p = 0.022

#touch_freq
mean_freq_df_prima = df_prima.groupby('code')['filt_touch_density_trial'].mean()
mean_freq_df_dopo = df_dopo.groupby('code')['filt_touch_density_trial'].mean()
pg.normality(mean_freq_df_prima)
pg.normality(mean_freq_df_dopo)
stats.levene(mean_freq_df_prima,mean_freq_df_dopo)
stats.ttest_ind(mean_freq_df_prima,mean_freq_df_dopo, equal_var=True) #p = 0.90

#amount_rotation
mean_rot_df_prima = df_prima.groupby('code')['rot_abs_total'].mean()
mean_rot_df_dopo = df_dopo.groupby('code')['rot_abs_total'].mean()
pg.normality(mean_rot_df_prima)
pg.normality(mean_rot_df_dopo)
stats.levene(mean_rot_df_prima,mean_rot_df_dopo)
stats.ttest_ind(mean_rot_df_prima,mean_rot_df_dopo, equal_var=True) #p = 0.91

#rotation vel
mean_rotv_df_prima = df_prima.groupby('code')['filtered_mean_rot_velocity'].mean()
mean_rotv_df_dopo = df_dopo.groupby('code')['filtered_mean_rot_velocity'].mean()
pg.normality(mean_rotv_df_prima)
pg.normality(mean_rotv_df_dopo)
stats.levene(mean_rotv_df_prima,mean_rotv_df_dopo)
stats.ttest_ind(mean_rotv_df_prima,mean_rotv_df_dopo, equal_var=True) #p = 0.123

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# correlation between accuracy and n° diff transitions
corr = stats.pointbiserialr(df['accuracy'], df['diff_trans'])

w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.boxplot(x=df['accuracy'],y=df['diff_trans'],showmeans=True)
ax.set(xlabel='Response', ylabel='Number of Different Transitions', title='Number of Different Transitions & Accuracy')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#check if the two clusters differ in n° diff transitions
bravi = df[df['cluster']==0]
scarsi = df[df['cluster']==1]

trans_bravi = bravi['diff_trans'].values
trans_scarsi = scarsi['diff_trans'].values

pg.normality(trans_bravi)
pg.normality(trans_scarsi)

stats.levene(trans_bravi,trans_scarsi)
stats.mannwhitneyu(trans_bravi,trans_scarsi,alternative='less')

w = 4
h = 3
d = 300
plt.figure(figsize=(w,h),dpi=d)
ax = sns.boxplot(x=df['cluster'],y=df['diff_trans'],showmeans=True)
ax.set(xlabel='Cluster', ylabel='Number of Different Transitions', title='Mean Number of Different Transitions by Cluster')