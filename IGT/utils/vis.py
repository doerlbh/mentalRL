from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

def getWinner(scores, names):
    scores = np.array(scores)
    names = np.array(names)
    return names[np.argmax(scores)]

# plot graphs of different RL algorithms
def plotAgents(reports,names,nTrials,reward_functions,labels,fig_name,is_flipped=False,isIGT=False,plotShortTerm=False,shortTerm=100):
    plt.rcParams['figure.figsize'] = [20, 10]
    lw = 2
    tmp = reports[0]['percent']
    steps = np.arange(tmp.shape[1])+1 
    fig = plt.figure()    
    fig.subplots_adjust(hspace=0.5)
    alpha_plot = 0.2
    data = []
    colors = ('black','blue','orange','red','green','purple','grey','gold','brown','navy')

    if isIGT:

        reward_from_A = reward_functions[0]
        reward_from_B = reward_functions[1]
        reward_from_C = reward_functions[2]
        reward_from_D = reward_functions[3]
        y,a1,b1,c1,d1,a2,b2,c2,d2,r,scores,rewards,actions = [],[],[],[],[],[],[],[],[],[],[],[],[]
        for rep in reports:
            if is_flipped: y.append(rep['percent'])
            else: y.append(100-rep['percent'])
            a1.append(rep["Q1(I)a"])
            b1.append(rep["Q1(I)b"])
            c1.append(rep["Q1(I)c"])
            d1.append(rep["Q1(I)d"])
            a2.append(rep["Q2(I)a"])
            b2.append(rep["Q2(I)b"])
            c2.append(rep["Q2(I)c"])
            d2.append(rep["Q2(I)d"])
            r.append(np.cumsum(rep["reward"],1))
            scores.append(np.mean(np.cumsum(rep["reward"],1),0)[-1])
            rewards.append(rep["reward"])
            actions.append(rep["actions"])

        if plotShortTerm:
            ax1 = plt.subplot2grid((3, 12), (0, 0), colspan=4)
            ax2 = plt.subplot2grid((3, 12), (0, 4), colspan=3)
            ax2b = plt.subplot2grid((3, 12), (0, 7), colspan=1)
            ax3 = plt.subplot2grid((3, 12), (0, 8), colspan=3)
            ax3b = plt.subplot2grid((3, 12), (0, 11), colspan=1)
            ax4 = plt.subplot2grid((3, 12), (1, 0), colspan=3)
            ax5 = plt.subplot2grid((3, 12), (1, 3), colspan=3)
            ax6 = plt.subplot2grid((3, 12), (1, 6), colspan=3)
            ax7 = plt.subplot2grid((3, 12), (1, 9), colspan=3)
            ax8 = plt.subplot2grid((3, 12), (2, 0), colspan=3)
            ax9 = plt.subplot2grid((3, 12), (2, 3), colspan=3)
            ax10 = plt.subplot2grid((3, 12), (2, 6), colspan=3)
            ax11 = plt.subplot2grid((3, 12), (2, 9), colspan=3)
            axlist=[[ax1,ax2,ax2b,ax3,ax3b],[ax4,ax5,ax6,ax7],[ax8,ax9,ax10,ax11]]        
        else:
            ax1 = plt.subplot2grid((3, 12), (0, 0), colspan=4)
            ax2 = plt.subplot2grid((3, 12), (0, 4), colspan=4)
            ax3 = plt.subplot2grid((3, 12), (0, 8), colspan=4)
            ax4 = plt.subplot2grid((3, 12), (1, 0), colspan=3)
            ax5 = plt.subplot2grid((3, 12), (1, 3), colspan=3)
            ax6 = plt.subplot2grid((3, 12), (1, 6), colspan=3)
            ax7 = plt.subplot2grid((3, 12), (1, 9), colspan=3)
            ax8 = plt.subplot2grid((3, 12), (2, 0), colspan=3)
            ax9 = plt.subplot2grid((3, 12), (2, 3), colspan=3)
            ax10 = plt.subplot2grid((3, 12), (2, 6), colspan=3)
            ax11 = plt.subplot2grid((3, 12), (2, 9), colspan=3)
            axlist=[[ax1,ax2,ax3],[ax4,ax5,ax6,ax7],[ax8,ax9,ax10,ax11]]

        for i in range(1000):
            a = np.sum(reward_from_A(i))
            b = np.sum(reward_from_B(i))
            c = np.sum(reward_from_C(i))
            d = np.sum(reward_from_D(i))
            data.append([a,b,c,d])
        data = pd.DataFrame(data, columns=labels)
        sns.kdeplot(data['A'],shade=True,ax=axlist[0][0])
        sns.kdeplot(data['B'],shade=True,ax=axlist[0][0])
        sns.kdeplot(data['C'],shade=True,ax=axlist[0][0])
        sns.kdeplot(data['D'],shade=True,ax=axlist[0][0])
        axlist[0][0].axvline(x=np.mean(data['A']), color='blue', linestyle='--')
        axlist[0][0].axvline(x=np.mean(data['B']), color='orange', linestyle='--')
        axlist[0][0].axvline(x=np.mean(data['C']), color='green', linestyle='--')
        axlist[0][0].axvline(x=np.mean(data['D']), color='red', linestyle='--')
        axlist[0][0].set_xlabel('reward')
        axlist[0][0].set_ylabel('Reward distributions for four actions')
        axlist[0][0].legend(loc='upper center', bbox_to_anchor=(0.1, 0.8), ncol=1)
        axlist[0][0].grid(True)

        if plotShortTerm:
            last = 10
            for i, _ in enumerate(y):
                l, = axlist[0][1].plot(steps[:shortTerm], np.mean(y[i],0)[:shortTerm], marker='', color=colors[i], linewidth=lw, label=names[i])
                l, = axlist[0][2].plot(steps[-last:], np.mean(y[i],0)[-last:], marker='', color=colors[i], linewidth=lw, label=names[i])
                l, = axlist[0][3].plot(steps[:shortTerm], np.mean(r[i],0)[:shortTerm], marker='', color=colors[i], linewidth=lw, label=names[i])
                l, = axlist[0][4].plot(steps[-last:], np.mean(r[i],0)[-last:], marker='', color=colors[i], linewidth=lw, label=names[i])
                l = axlist[0][1].fill_between(steps[:shortTerm], np.mean(y[i],0)[:shortTerm]-np.std(y[i],0)[:shortTerm]/math.sqrt(nTrials),np.mean(y[i],0)[:shortTerm]+np.std(y[i],0)[:shortTerm]/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
                l = axlist[0][2].fill_between(steps[-last:], np.mean(y[i],0)[-last:]-np.std(y[i],0)[-last:]/math.sqrt(nTrials),np.mean(y[i],0)[-last:]+np.std(y[i],0)[-last:]/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
                l = axlist[0][3].fill_between(steps[:shortTerm], np.mean(r[i],0)[:shortTerm]-np.std(r[i],0)[:shortTerm]/math.sqrt(nTrials),np.mean(r[i],0)[:shortTerm]+np.std(r[i],0)[:shortTerm]/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
                l = axlist[0][4].fill_between(steps[-last:], np.mean(r[i],0)[-last:]-np.std(r[i],0)[-last:]/math.sqrt(nTrials),np.mean(r[i],0)[-last:]+np.std(r[i],0)[-last:]/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
            axlist[0][1].set_xlabel('Episodes')
            axlist[0][1].set_ylabel('% choosing better decks (short-term)')
            axlist[0][1].set_ylim(0,100)
            axlist[0][1].grid(True)
#             axlist[0][2].set_xlabel('Episodes')
#             axlist[0][2].get_yaxis().set_visible(False)
#             axlist[0][2].yaxis('off')
#             axlist[0][2].yaxis.tick_right()
            axlist[0][2].set_ylabel('% choosing better decks (long-term)')
            axlist[0][2].set_ylim(0,100)
            axlist[0][2].grid(True)
            axlist[0][3].set_xlabel('Episodes')
            axlist[0][3].set_ylabel('Cumulative episode rewards')
            axlist[0][3].legend()
            axlist[0][3].grid(True)
        else:
            for i, _ in enumerate(y):
                l, = axlist[0][1].plot(steps, np.mean(y[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
                l, = axlist[0][2].plot(steps, np.mean(r[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
                l = axlist[0][1].fill_between(steps, np.mean(y[i],0)-np.std(y[i],0)/math.sqrt(nTrials),np.mean(y[i],0)+np.std(y[i],0)/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
                l = axlist[0][2].fill_between(steps, np.mean(r[i],0)-np.std(r[i],0)/math.sqrt(nTrials),np.mean(r[i],0)+np.std(r[i],0)/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
            axlist[0][1].set_xlabel('Episodes')
            axlist[0][1].set_ylabel('% choosing better decks')
            axlist[0][1].set_ylim(0,100)
            axlist[0][1].grid(True)
            axlist[0][2].set_xlabel('Episodes')
            axlist[0][2].set_ylabel('Cumulative episode rewards')
            axlist[0][2].grid(True)

        for i, _ in enumerate(y):
            l, = axlist[1][0].plot(steps, np.mean(a1[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l, = axlist[1][1].plot(steps, np.mean(b1[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l, = axlist[1][2].plot(steps, np.mean(c1[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l, = axlist[1][3].plot(steps, np.mean(d1[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l, = axlist[2][0].plot(steps, np.mean(a2[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l, = axlist[2][1].plot(steps, np.mean(b2[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l, = axlist[2][2].plot(steps, np.mean(c2[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l, = axlist[2][3].plot(steps, np.mean(d2[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l = axlist[1][0].fill_between(steps, np.mean(a1[i],0)-np.std(a1[i],0)/math.sqrt(nTrials),np.mean(a1[i],0)+np.std(a1[i],0)/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
            l = axlist[1][1].fill_between(steps, np.mean(b1[i],0)-np.std(b1[i],0)/math.sqrt(nTrials),np.mean(b1[i],0)+np.std(b1[i],0)/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
            l = axlist[1][2].fill_between(steps, np.mean(c1[i],0)-np.std(c1[i],0)/math.sqrt(nTrials),np.mean(c1[i],0)+np.std(c1[i],0)/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
            l = axlist[1][3].fill_between(steps, np.mean(d1[i],0)-np.std(d1[i],0)/math.sqrt(nTrials),np.mean(d1[i],0)+np.std(d1[i],0)/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
            l = axlist[2][0].fill_between(steps, np.mean(a2[i],0)-np.std(a2[i],0)/math.sqrt(nTrials),np.mean(a2[i],0)+np.std(a2[i],0)/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
            l = axlist[2][1].fill_between(steps, np.mean(b2[i],0)-np.std(b2[i],0)/math.sqrt(nTrials),np.mean(b2[i],0)+np.std(b2[i],0)/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
            l = axlist[2][2].fill_between(steps, np.mean(c2[i],0)-np.std(c2[i],0)/math.sqrt(nTrials),np.mean(c2[i],0)+np.std(c2[i],0)/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])
            l = axlist[2][3].fill_between(steps, np.mean(d2[i],0)-np.std(d2[i],0)/math.sqrt(nTrials),np.mean(d2[i],0)+np.std(d2[i],0)/math.sqrt(nTrials),alpha=alpha_plot,color=colors[i])

        axlist[1][0].set_xlabel('Episodes')
        axlist[1][0].set_ylabel('Q1 for picking A')
        axlist[1][0].grid(True)
        axlist[1][1].set_xlabel('Episodes')
        axlist[1][1].set_ylabel('Q1 for picking B')
        axlist[1][1].grid(True)
        axlist[1][2].set_xlabel('Episodes')
        axlist[1][2].set_ylabel('Q1 for picking C')
        axlist[1][2].grid(True)
        axlist[1][3].set_xlabel('Episodes')
        axlist[1][3].set_ylabel('Q1 for picking D')
        axlist[1][3].grid(True)
        axlist[2][0].set_xlabel('Episodes')
        axlist[2][0].set_ylabel('Q2 for picking A')
        axlist[2][0].grid(True)
        axlist[2][1].set_xlabel('Episodes')
        axlist[2][1].set_ylabel('Q2 for picking B')
        axlist[2][1].grid(True)
        axlist[2][2].set_xlabel('Episodes')
        axlist[2][2].set_ylabel('Q2 for picking C')
        axlist[2][2].grid(True)
        axlist[2][3].set_xlabel('Episodes')
        axlist[2][3].set_ylabel('Q2 for picking D')
        axlist[2][3].legend()
        axlist[2][3].grid(True) 
        
        fig.tight_layout(pad=0., w_pad=0., h_pad=0.5) 

    else:

        reward_from_B,reward_from_C = reward_functions[0],reward_functions[1]
        label_B,label_C = labels[0],labels[1]
        y,l1,r1,l2,r2,c,scores,rewards,actions = [],[],[],[],[],[],[],[],[]
        for rep in reports:
            if is_flipped: y.append(rep['percent'])
            else: y.append(100-rep['percent'])
            l1.append(rep["Q1(A)l"])
            r1.append(rep["Q1(A)r"])
            l2.append(rep["Q2(A)l"])
            r2.append(rep["Q2(A)r"])
            c.append(np.cumsum(rep["reward"],1))
            scores.append(np.mean(np.cumsum(rep["reward"],1),0)[-1])
            rewards.append(rep["reward"])
            actions.append(rep["actions"])

        ax1 = plt.subplot2grid((2, 12), (0, 0), colspan=4)
        ax2 = plt.subplot2grid((2, 12), (0, 4), colspan=4)
        ax3 = plt.subplot2grid((2, 12), (0, 8), colspan=4)
        ax4 = plt.subplot2grid((2, 12), (1, 0), colspan=3)
        ax5 = plt.subplot2grid((2, 12), (1, 3), colspan=3)
        ax6 = plt.subplot2grid((2, 12), (1, 6), colspan=3)
        ax7 = plt.subplot2grid((2, 12), (1, 9), colspan=3)
        axlist=[[ax1,ax2,ax3],[ax4,ax5,ax6,ax7]]
    
        for i in range(5000): data.append([reward_from_B(),reward_from_C()])
        data = pd.DataFrame(data, columns=labels)
        sns.kdeplot(data[label_B],shade=True,ax=axlist[0][0])
        sns.kdeplot(data[label_C],shade=True,ax=axlist[0][0])
        axlist[0][0].axvline(x=np.mean(data[label_B]), color='blue', linestyle='--')
        axlist[0][0].axvline(x=np.mean(data[label_C]), color='orange', linestyle='--')
        axlist[0][0].set_xlabel('reward')
        axlist[0][0].set_ylabel('Reward distributions for action left vs. right')
    
        for i, _ in enumerate(y):
            l, = axlist[0][1].plot(steps, np.mean(y[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l  = axlist[0][1].fill_between(steps, np.mean(y[i],0)-np.std(y[i],0)/math.sqrt(nTrials),np.mean(y[i],0)+np.std(y[i],0)/math.sqrt(nTrials),alpha=alpha_plot, color=colors[i])
            l, = axlist[0][2].plot(steps, np.mean(c[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l  = axlist[0][2].fill_between(steps, np.mean(c[i],0)-np.std(c[i],0)/math.sqrt(nTrials),np.mean(c[i],0)+np.std(c[i],0) /math.sqrt(nTrials),alpha=alpha_plot, color=colors[i])
            l, = axlist[1][0].plot(steps, np.mean(l1[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l  = axlist[1][0].fill_between(steps, np.mean(l1[i],0)-np.std(l1[i],0)/math.sqrt(nTrials),np.mean(l1[i],0)+np.std(l1[i],0)/math.sqrt(nTrials),alpha=alpha_plot, color=colors[i])
            l, = axlist[1][1].plot(steps, np.mean(r1[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l  = axlist[1][1].fill_between(steps, np.mean(r1[i],0)-np.std(r1[i],0)/math.sqrt(nTrials),np.mean(r1[i],0)+np.std(r1[i],0)/math.sqrt(nTrials),alpha=alpha_plot, color=colors[i])
            l, = axlist[1][2].plot(steps, np.mean(l2[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l  = axlist[1][2].fill_between(steps, np.mean(l2[i],0)-np.std(l2[i],0)/math.sqrt(nTrials),np.mean(l2[i],0)+np.std(l2[i],0)/math.sqrt(nTrials),alpha=alpha_plot, color=colors[i])
            l, = axlist[1][3].plot(steps, np.mean(r2[i],0) , marker='', color=colors[i], linewidth=lw, label=names[i])
            l  = axlist[1][3].fill_between(steps, np.mean(r2[i],0)-np.std(r2[i],0)/math.sqrt(nTrials),np.mean(r2[i],0)+np.std(r2[i],0)/math.sqrt(nTrials),alpha=alpha_plot, color=colors[i])
        axlist[0][1].set_xlabel('Episodes')
        axlist[0][1].set_ylabel('% choosing better action')
        axlist[0][1].set_ylim(0,100)
        axlist[0][1].grid(True)
        axlist[0][2].set_xlabel('Episodes')
        axlist[0][2].set_ylabel('Cumulative episode rewards')
        axlist[0][2].grid(True)
        axlist[1][0].set_xlabel('Episodes')
        axlist[1][0].set_ylabel('Q1 for action left at state A')
        axlist[1][0].grid(True)
        axlist[1][1].set_xlabel('Episodes')
        axlist[1][1].set_ylabel('Q1 for action right at state A')
#         axlist[1][1].legend()
        axlist[1][1].grid(True)
        axlist[1][2].set_xlabel('Episodes')
        axlist[1][2].set_ylabel('Q2 for action left at state A')
        axlist[1][2].grid(True)
        axlist[1][3].set_xlabel('Episodes')
        axlist[1][3].set_ylabel('Q2 for action right at state A')
        axlist[1][3].legend()
        axlist[1][3].grid(True)
    
        fig.tight_layout() 
        
    fig.savefig(fig_name)
    
    return fig, scores,rewards,actions
 