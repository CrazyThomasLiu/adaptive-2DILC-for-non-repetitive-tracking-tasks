import copy
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
save_figure=False
iteration_num=20
#batch_explore=25


"1. load RMSE of mbilc from the csv"
df_mbilc = pd.read_csv('./Data/RMSE_mbilc.csv', index_col=0)
RMSE_mbilc = df_mbilc.to_numpy()
RMSE_mbilc=RMSE_mbilc[10:iteration_num]
print('mbilc:')
print('mean',np.mean(RMSE_mbilc))
print('max',np.max(RMSE_mbilc))
print('min',np.min(RMSE_mbilc))
#pdb.set_trace()
"2. load RMSE of mfilc from the csv"
df_mfilc = pd.read_csv('./Data/RMSE_mfilc.csv', index_col=0)
RMSE_mfilc = df_mfilc.to_numpy()
RMSE_mfilc=RMSE_mfilc[10:iteration_num]

print('mfilc:')
print('mean',np.mean(RMSE_mfilc))
print('max',np.max(RMSE_mfilc))
print('min',np.min(RMSE_mfilc))

"3. load RMSE of PI robust ILC from the csv"
df_pi_robust = pd.read_csv('./Data/RMSE_PI_Robust.csv', index_col=0)
RMSE_pi_robust = df_pi_robust.to_numpy()
RMSE_pi_robust=RMSE_pi_robust[10:iteration_num]

print('pi_robust:')
print('mean',np.mean(RMSE_pi_robust))
print('max',np.max(RMSE_pi_robust))
print('min',np.min(RMSE_pi_robust))
pdb.set_trace()


'3 Plot figure'
iteration_axis=range(1,iteration_num+1)
plt.rcParams['pdf.fonttype'] = 42
fig,ax0=plt.subplots(1,1,figsize=(8,5.5))
#fig4=plt.figure(figsize=(7,5.5))
x_major_locator=MultipleLocator(int(iteration_num/10))
ax0=plt.gca()
ax0.xaxis.set_major_locator(x_major_locator)
ax0.plot(iteration_axis,RMSE_mbilc,linewidth=2,color='darkorange',linestyle = 'solid')
ax0.plot(iteration_axis,RMSE_mfilc,linewidth=2.3,color='royalblue',linestyle = 'dotted')
ax0.plot(iteration_axis,RMSE_pi_robust,linewidth=2,color='orangered',linestyle = 'dashdot')
ax0.grid()

xlable = 'Batch:$k$'
ylable = 'RMSE'
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 16,
         }
#ax0.set_ylabel('$\| \pi^{i}-\pi^{*} \|$',font2)
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.tick_params(labelsize=13)
plt.legend(['2D model-based ILC','2D model-free adaptive ILC','Robust Pi-based ILC'],fontsize='13')
if save_figure==True:
    #plt.savefig('RMSE_comparation.jpg',dpi=800)
    plt.savefig('RMSE_comparation.pdf')
plt.show()
