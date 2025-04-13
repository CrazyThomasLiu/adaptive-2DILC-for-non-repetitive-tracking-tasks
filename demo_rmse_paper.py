import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator
save_figure=True
iteration_num=20


"1. load RMSE of mbilc from the csv"
df_mbilc = pd.read_csv('./Data/RMSE_mbilc_paper.csv', index_col=0)
RMSE_mbilc = df_mbilc.to_numpy()
RMSE_mbilc=RMSE_mbilc[0:iteration_num]

"2. load RMSE of mfilc from the csv"
df_mfilc = pd.read_csv('./Data/RMSE_mfilc_paper.csv', index_col=0)
RMSE_mfilc = df_mfilc.to_numpy()
RMSE_mfilc=RMSE_mfilc[0:iteration_num]



"3. load RMSE of PI robust ILC from the csv"
df_pi_robust = pd.read_csv('./Data/RMSE_PI_Robust_paper.csv', index_col=0)
RMSE_pi_robust = df_pi_robust.to_numpy()
RMSE_pi_robust=RMSE_pi_robust[0:iteration_num]




'3 Plot figure'
iteration_axis=range(1,iteration_num+1)
plt.rcParams['pdf.fonttype'] = 42
fig,ax0=plt.subplots(1,1,figsize=(8,5.5))
x_major_locator=MultipleLocator(int(iteration_num/10))
ax0=plt.gca()
ax0.xaxis.set_major_locator(x_major_locator)
ax0.plot(iteration_axis,RMSE_mbilc,linewidth=2,color='darkorange',linestyle = 'solid')
ax0.plot(iteration_axis,RMSE_mfilc,linewidth=2.3,color='royalblue',linestyle = 'dotted')
ax0.plot(iteration_axis,RMSE_pi_robust,linewidth=2,color='orangered',linestyle = 'dashdot')
ax0.grid()

xlable = 'Batch: $k$'
ylable = 'RMSE'
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 16,
         }
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.tick_params(labelsize=13)
plt.legend(['Model-based 2D ILC','Model-free adaptive 2D ILC','Robust PI-based ILC'],fontsize='13')
if save_figure==True:
    plt.savefig('RMSE_paper.pdf')
plt.show()
