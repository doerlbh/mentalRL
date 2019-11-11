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
    
# draw graphs of different RL algorithms
def drawGraph(reportQl, reportDoubleQl, reportSplitQl,reportPositiveQl,reportNegativeQl,reportExponentialDoubleQl,reportExponentialSplitQl,reportSARSA,reward_from_B,reward_from_C,label_B,label_C,fig_name,draw_Exponential=False,draw_SARSA=False,is_flipped=False):
    plt.rcParams['figure.figsize'] = [20, 10]
    lw = 2
    
    yQL,yDQL,ySQL,yPQL,yNQL,yEDQL,yESQL,ySARSA = reportQl['percent'],reportDoubleQl['percent'],reportSplitQl['percent'],reportPositiveQl['percent'],reportNegativeQl['percent'],reportExponentialDoubleQl['percent'],reportExponentialSplitQl['percent'],reportSARSA['percent']
    if not is_flipped:
        yQL,yDQL,ySQL,yPQL,yNQL,yEDQL,yESQL,ySARSA = 100-yQL,100-yDQL,100-ySQL,100-yPQL,100-yNQL,100-yEDQL,100-yESQL,100-ySARSA
    lQL1,lDQL1,lSQL1,lPQL1,lNQL1,lEDQL1,lESQL1,lSARSA1 = reportQl["Q1(A)l"],reportDoubleQl["Q1(A)l"],reportSplitQl["Q1(A)l"],reportPositiveQl["Q1(A)l"],reportNegativeQl["Q1(A)l"],reportExponentialDoubleQl["Q1(A)l"],reportExponentialSplitQl["Q1(A)l"],reportSARSA["Q1(A)l"]
    rQL1,rDQL1,rSQL1,rPQL1,rNQL1,rEDQL1,rESQL1,rSARSA1 = reportQl["Q1(A)r"],reportDoubleQl["Q1(A)r"],reportSplitQl["Q1(A)r"],reportPositiveQl["Q1(A)r"],reportNegativeQl["Q1(A)r"],reportExponentialDoubleQl["Q1(A)r"],reportExponentialSplitQl["Q1(A)r"],reportSARSA["Q1(A)r"]
    lQL2,lDQL2,lSQL2,lPQL2,lNQL2,lEDQL2,lESQL2,lSARSA2 = reportQl["Q1(A)l"],reportDoubleQl["Q2(A)l"],reportSplitQl["Q2(A)l"],reportPositiveQl["Q2(A)l"],reportNegativeQl["Q2(A)l"],reportExponentialDoubleQl["Q2(A)l"],reportExponentialSplitQl["Q2(A)l"],reportSARSA["Q1(A)l"]
    rQL2,rDQL2,rSQL2,rPQL2,rNQL2,rEDQL2,rESQL2,rSARSA2 = reportQl["Q1(A)r"],reportDoubleQl["Q2(A)r"],reportSplitQl["Q2(A)r"],reportPositiveQl["Q2(A)r"],reportNegativeQl["Q2(A)r"],reportExponentialDoubleQl["Q2(A)r"],reportExponentialSplitQl["Q2(A)r"],reportSARSA["Q1(A)r"]
    cQL,cDQL,cSQL,cPQL,cNQL,cEDQL,cESQL,cSARSA = np.cumsum(reportQl["reward"],1),np.cumsum(reportDoubleQl["reward"],1),np.cumsum(reportSplitQl["reward"],1),np.cumsum(reportPositiveQl["reward"],1),np.cumsum(reportNegativeQl["reward"],1),np.cumsum(reportExponentialDoubleQl["reward"],1),np.cumsum(reportExponentialSplitQl["reward"],1),np.cumsum(reportSARSA["reward"],1)
    
    steps = np.arange(yQL.shape[1])+1
      
    fig = plt.figure()    
    fig.subplots_adjust(hspace=0.5)

    ax1 = plt.subplot2grid((2, 12), (0, 0), colspan=4)
    ax2 = plt.subplot2grid((2, 12), (0, 4), colspan=4)
    ax3 = plt.subplot2grid((2, 12), (0, 8), colspan=4)
    ax4 = plt.subplot2grid((2, 12), (1, 0), colspan=3)
    ax5 = plt.subplot2grid((2, 12), (1, 3), colspan=3)
    ax6 = plt.subplot2grid((2, 12), (1, 6), colspan=3)
    ax7 = plt.subplot2grid((2, 12), (1, 9), colspan=3)
    axlist=[[ax1,ax2,ax3],[ax4,ax5,ax6,ax7]]
    
    data = []
    for i in range(5000):
        data.append([reward_from_B(),reward_from_C()])
    data = pd.DataFrame(data, columns=[label_B, label_C])
    sns.kdeplot(data[label_B],shade=True,ax=axlist[0][0])
    sns.kdeplot(data[label_C],shade=True,ax=axlist[0][0])
    axlist[0][0].axvline(x=np.mean(data[label_B]), color='blue', linestyle='--')
    axlist[0][0].axvline(x=np.mean(data[label_C]), color='orange', linestyle='--')
    axlist[0][0].set_xlabel('reward')
    axlist[0][0].set_ylabel('Reward distributions for action left vs. right')

    l_ql, = axlist[0][1].plot(steps, np.mean(yQL,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[0][1].plot(steps, np.mean(yDQL,0), marker='', color='blue',linewidth=lw, label="DQL")
    l_sql, = axlist[0][1].plot(steps, np.mean(ySQL,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[0][1].plot(steps, np.mean(yPQL,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[0][1].plot(steps, np.mean(yNQL,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[0][1].plot(steps, np.mean(ySARSA,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[0][1].plot(steps, np.mean(yEDQL,0), marker='', color='purple',linewidth=lw, label="EDQL")
        l_esql, = axlist[0][1].plot(steps, np.mean(yESQL,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[0][1].set_xlabel('Episodes')
    axlist[0][1].set_ylabel('% choosing better action')
    axlist[0][1].set_ylim(0,100)
    axlist[0][1].grid(True)

    l_ql, = axlist[0][2].plot(steps, np.mean(cQL,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[0][2].plot(steps, np.mean(cDQL,0), marker='', color='blue',linewidth=lw, label="DQL")
    l_sql, = axlist[0][2].plot(steps, np.mean(cSQL,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[0][2].plot(steps, np.mean(cPQL,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[0][2].plot(steps, np.mean(cNQL,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[0][2].plot(steps, np.mean(cSARSA,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[0][2].plot(steps, np.mean(cEDQL,0), marker='', color='purple',linewidth=lw, label="EDQL")
        l_esql, = axlist[0][2].plot(steps, np.mean(cESQL,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[0][2].set_xlabel('Episodes')
    axlist[0][2].set_ylabel('Cumulative episode rewards')
    axlist[0][2].grid(True)
    
    l_ql, = axlist[1][0].plot(steps, np.mean(lQL1,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[1][0].plot(steps, np.mean(lDQL1,0), marker='', color='blue',linewidth=lw, label="DQL")
    l_sql, = axlist[1][0].plot(steps, np.mean(lSQL1,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[1][0].plot(steps, np.mean(lPQL1,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[1][0].plot(steps, np.mean(lNQL1,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[1][0].plot(steps, np.mean(lSARSA1,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[1][0].plot(steps, np.mean(lEDQL1,0), marker='', color='purple',linewidth=lw, label="EDQL")
        l_esql, = axlist[1][0].plot(steps, np.mean(lESQL1,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[1][0].set_xlabel('Episodes')
    axlist[1][0].set_ylabel('Q1 for action left at state A')
    axlist[1][0].grid(True)

    l_ql, = axlist[1][1].plot(steps, np.mean(rQL1,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[1][1].plot(steps, np.mean(rDQL1,0), marker='', color='blue',linewidth=lw, label="DQL")
    l_sql, = axlist[1][1].plot(steps, np.mean(rSQL1,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[1][1].plot(steps, np.mean(rPQL1,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[1][1].plot(steps, np.mean(rNQL1,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[1][1].plot(steps, np.mean(rSARSA1,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[1][1].plot(steps, np.mean(rEDQL1,0), marker='', color='purple',linewidth=lw, label="EDQL")
        l_esql, = axlist[1][1].plot(steps, np.mean(rESQL1,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[1][1].set_xlabel('Episodes')
    axlist[1][1].set_ylabel('Q1 for action right at state A')
#     axlist[1][1].legend()
    axlist[1][1].grid(True)

    l_ql, = axlist[1][2].plot(steps, np.mean(lQL2,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[1][2].plot(steps, np.mean(lDQL2,0), marker='', color='blue', linewidth=lw, label="DQL")
    l_sql, = axlist[1][2].plot(steps, np.mean(lSQL2,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[1][2].plot(steps, np.mean(lPQL2,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[1][2].plot(steps, np.mean(lNQL2,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[1][2].plot(steps, np.mean(lSARSA2,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[1][2].plot(steps, np.mean(lEDQL2,0), marker='', color='purple', linewidth=lw, label="EDQL")
        l_esql, = axlist[1][2].plot(steps, np.mean(lESQL2,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[1][2].set_xlabel('Episodes')
    axlist[1][2].set_ylabel('Q2 for action left at state A')
    axlist[1][2].grid(True)

    l_ql, = axlist[1][3].plot(steps, np.mean(rQL2,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[1][3].plot(steps, np.mean(rDQL2,0), marker='', color='blue', linewidth=lw, label="DQL")
    l_sql, = axlist[1][3].plot(steps, np.mean(rSQL2,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[1][3].plot(steps, np.mean(rPQL2,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[1][3].plot(steps, np.mean(rNQL2,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[1][3].plot(steps, np.mean(rSARSA2,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[1][3].plot(steps, np.mean(rEDQL2,0), marker='', color='purple', linewidth=lw, label="EDQL")
        l_esql, = axlist[1][3].plot(steps, np.mean(rESQL2,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[1][3].set_xlabel('Episodes')
    axlist[1][3].set_ylabel('Q2 for action right at state A')
    axlist[1][3].legend()
    axlist[1][3].grid(True)
    
    alpha_plot = 0.2
    
    l_ql  = axlist[0][1].fill_between(steps, np.mean(yQL,0)-np.std(yQL,0)/math.sqrt(cntExperiments),np.mean(yQL,0)+np.std(yQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_dql = axlist[0][1].fill_between(steps, np.mean(yDQL,0)-np.std(yDQL,0)/math.sqrt(cntExperiments),np.mean(yDQL,0)+np.std(yDQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_sql = axlist[0][1].fill_between(steps, np.mean(ySQL,0)-np.std(ySQL,0)/math.sqrt(cntExperiments),np.mean(ySQL,0)+np.std(ySQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_pql = axlist[0][1].fill_between(steps, np.mean(yPQL,0)-np.std(yPQL,0)/math.sqrt(cntExperiments),np.mean(yPQL,0)+np.std(yPQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_nql = axlist[0][1].fill_between(steps, np.mean(yNQL,0)-np.std(yNQL,0)/math.sqrt(cntExperiments),np.mean(yNQL,0)+np.std(yNQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    if draw_SARSA:
        l_sarsa = axlist[0][1].fill_between(steps,np.mean(ySARSA,0)-np.mean(ySARSA,0)/math.sqrt(cntExperiments),np.mean(ySARSA,0)+np.mean(ySARSA,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='gold')
    if draw_Exponential:
        l_edql = axlist[0][1].fill_between(steps,np.mean(yEDQL,0)-np.mean(yEDQL,0)/math.sqrt(cntExperiments),np.mean(yEDQL,0)+np.mean(yEDQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
        l_esql = axlist[0][1].fill_between(steps,np.mean(yESQL,0)-np.mean(yESQL,0)/math.sqrt(cntExperiments),np.mean(yESQL,0)+np.mean(yESQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')

    l_ql = axlist[0][2].fill_between(steps, np.mean(cQL,0)-np.std(cQL,0)/math.sqrt(cntExperiments),np.mean(cQL,0)+np.std(cQL,0) /math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_dql = axlist[0][2].fill_between(steps, np.mean(cDQL,0)-np.std(cDQL,0)/math.sqrt(cntExperiments),np.mean(cDQL,0)+np.std(cDQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_sql = axlist[0][2].fill_between(steps, np.mean(cSQL,0)-np.std(cSQL,0)/math.sqrt(cntExperiments),np.mean(cSQL,0)+np.std(cSQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_pql = axlist[0][2].fill_between(steps, np.mean(cPQL,0)-np.std(cPQL,0)/math.sqrt(cntExperiments),np.mean(cPQL,0)+np.std(cPQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_nql = axlist[0][2].fill_between(steps, np.mean(cNQL,0)-np.std(cNQL,0)/math.sqrt(cntExperiments),np.mean(cNQL,0)+np.std(cNQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    if draw_SARSA:
        l_sarsa = axlist[0][2].fill_between(steps,np.mean(cSARSA,0)-np.mean(cSARSA,0)/math.sqrt(cntExperiments),np.mean(cSARSA,0)+np.mean(cSARSA,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='gold')
    if draw_Exponential:
        l_edql = axlist[0][2].fill_between(steps,np.mean(cEDQL,0)-np.mean(cEDQL,0)/math.sqrt(cntExperiments),np.mean(cEDQL,0)+np.mean(cEDQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
        l_esql = axlist[0][2].fill_between(steps,np.mean(cESQL,0)-np.mean(cESQL,0)/math.sqrt(cntExperiments),np.mean(cESQL,0)+np.mean(cESQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')
    
    l_ql = axlist[1][0].fill_between(steps, np.mean(lQL1,0)-np.std(lQL1,0)/math.sqrt(cntExperiments),np.mean(lQL1,0)-np.std(lQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_dql = axlist[1][0].fill_between(steps, np.mean(lDQL1,0)-np.std(lDQL1,0)/math.sqrt(cntExperiments),np.mean(lDQL1,0)+np.std(lDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_sql = axlist[1][0].fill_between(steps, np.mean(lSQL1,0)-np.std(lSQL1,0)/math.sqrt(cntExperiments),np.mean(lSQL1,0)+np.std(lSQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_pql = axlist[1][0].fill_between(steps, np.mean(lPQL1,0)-np.std(lPQL1,0)/math.sqrt(cntExperiments),np.mean(lPQL1,0)+np.std(lPQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_nql = axlist[1][0].fill_between(steps, np.mean(lNQL1,0)-np.std(lNQL1,0)/math.sqrt(cntExperiments),np.mean(lNQL1,0)+np.std(lNQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    if draw_SARSA:
        l_sarsa = axlist[1][0].fill_between(steps,np.mean(lSARSA1,0)-np.mean(lSARSA1,0)/math.sqrt(cntExperiments),np.mean(lSARSA1,0)+np.mean(lSARSA1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='gold')
    if draw_Exponential:
        l_edql = axlist[1][0].fill_between(steps,np.mean(lEDQL1,0)-np.mean(lEDQL1,0)/math.sqrt(cntExperiments),np.mean(lEDQL1,0)+np.mean(lEDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
        l_esql = axlist[1][0].fill_between(steps,np.mean(lESQL1,0)-np.mean(lESQL1,0)/math.sqrt(cntExperiments),np.mean(lESQL1,0)+np.mean(lESQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')

    l_ql = axlist[1][1].fill_between(steps, np.mean(rQL1,0)-np.std(rQL1,0)/math.sqrt(cntExperiments),np.mean(rQL1,0)+np.std(rQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_dql = axlist[1][1].fill_between(steps, np.mean(rDQL1,0)-np.std(rDQL1,0)/math.sqrt(cntExperiments),np.mean(rDQL1,0)+np.std(rDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_sql = axlist[1][1].fill_between(steps, np.mean(rSQL1,0)-np.std(rSQL1,0)/math.sqrt(cntExperiments),np.mean(rSQL1,0)+np.std(rSQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_pql = axlist[1][1].fill_between(steps, np.mean(rPQL1,0)-np.std(rPQL1,0)/math.sqrt(cntExperiments),np.mean(rPQL1,0)+np.std(rPQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_nql = axlist[1][1].fill_between(steps, np.mean(rNQL1,0)-np.std(rNQL1,0)/math.sqrt(cntExperiments),np.mean(rNQL1,0)+np.std(rNQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    if draw_SARSA:
        l_sarsa = axlist[1][1].fill_between(steps,np.mean(rSARSA1,0)-np.mean(lSARSA1,0)/math.sqrt(cntExperiments),np.mean(lSARSA1,0)+np.mean(lSARSA1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='gold')
    if draw_Exponential:
        l_edql = axlist[1][1].fill_between(steps,np.mean(rEDQL1,0)-np.mean(rEDQL1,0)/math.sqrt(cntExperiments),np.mean(rEDQL1,0)+np.mean(rEDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
        l_esql = axlist[1][1].fill_between(steps,np.mean(rESQL1,0)-np.mean(rESQL1,0)/math.sqrt(cntExperiments),np.mean(rESQL1,0)+np.mean(rESQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')

    l_ql = axlist[1][2].fill_between(steps, np.mean(lQL2,0)-np.std(lQL2,0)/math.sqrt(cntExperiments),np.mean(lQL2,0)+np.std(lQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_dql = axlist[1][2].fill_between(steps, np.mean(lDQL2,0)-np.std(lDQL2,0)/math.sqrt(cntExperiments),np.mean(lDQL2,0)+np.std(lDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_sql = axlist[1][2].fill_between(steps, np.mean(lSQL2,0)-np.std(lSQL2,0)/math.sqrt(cntExperiments),np.mean(lSQL2,0)+np.std(lSQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_pql = axlist[1][2].fill_between(steps, np.mean(lPQL2,0)-np.std(lPQL2,0)/math.sqrt(cntExperiments),np.mean(lPQL2,0)+np.std(lPQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_nql = axlist[1][2].fill_between(steps, np.mean(lNQL2,0)-np.std(lNQL2,0)/math.sqrt(cntExperiments),np.mean(lNQL2,0)+np.std(lNQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    if draw_SARSA:
        l_sarsa = axlist[1][2].fill_between(steps,np.mean(lSARSA2,0)-np.mean(lSARSA2,0)/math.sqrt(cntExperiments),np.mean(lSARSA2,0)+np.mean(lSARSA2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='gold')
    if draw_Exponential:
        l_edql = axlist[1][2].fill_between(steps,np.mean(lEDQL2,0)-np.mean(lEDQL2,0)/math.sqrt(cntExperiments),np.mean(lEDQL2,0)+np.mean(lEDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
        l_esql = axlist[1][2].fill_between(steps,np.mean(lESQL2,0)-np.mean(lESQL2,0)/math.sqrt(cntExperiments),np.mean(lESQL2,0)+np.mean(lESQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')

    l_ql = axlist[1][3].fill_between(steps, np.mean(rQL2,0)-np.std(rQL2,0)/math.sqrt(cntExperiments),np.mean(rQL2,0)+np.std(rQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_dql = axlist[1][3].fill_between(steps, np.mean(rDQL2,0)-np.std(rDQL2,0)/math.sqrt(cntExperiments),np.mean(rDQL2,0)+np.std(rDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_sql = axlist[1][3].fill_between(steps, np.mean(rSQL2,0)-np.std(rSQL2,0)/math.sqrt(cntExperiments),np.mean(rSQL2,0)+np.std(rSQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_pql = axlist[1][3].fill_between(steps, np.mean(rPQL2,0)-np.std(rPQL2,0)/math.sqrt(cntExperiments),np.mean(rPQL2,0)+np.std(rPQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_nql = axlist[1][3].fill_between(steps, np.mean(rNQL2,0)-np.std(rNQL2,0)/math.sqrt(cntExperiments),np.mean(rNQL2,0)+np.std(rNQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    if draw_SARSA:
        l_sarsa = axlist[1][3].fill_between(steps,np.mean(rSARSA2,0)-np.mean(lSARSA2,0)/math.sqrt(cntExperiments),np.mean(lSARSA2,0)+np.mean(lSARSA2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='gold')
    if draw_Exponential:
        l_edql = axlist[1][3].fill_between(steps,np.mean(rEDQL2,0)-np.mean(rEDQL2,0)/math.sqrt(cntExperiments),np.mean(rEDQL2,0)+np.mean(rEDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
        l_esql = axlist[1][3].fill_between(steps,np.mean(rESQL2,0)-np.mean(rESQL2,0)/math.sqrt(cntExperiments),np.mean(rESQL2,0)+np.mean(rESQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')

    fig.tight_layout() 
    fig.savefig(fig_name)
    
    return fig, np.mean(cQL,0)[-1],np.mean(cDQL,0)[-1],np.mean(cSQL,0)[-1],np.mean(cPQL,0)[-1],np.mean(cNQL,0)[-1],np.mean(cEDQL,0)[-1],np.mean(cESQL,0)[-1],np.mean(cSARSA,0)[-1],reportQl["reward"],reportDoubleQl["reward"],reportSplitQl["reward"],reportPositiveQl["reward"],reportNegativeQl["reward"],reportExponentialDoubleQl["reward"],reportExponentialSplitQl["reward"],reportSARSA["reward"],reportQl["actions"],reportDoubleQl["actions"],reportSplitQl["actions"],reportPositiveQl["actions"],reportNegativeQl["actions"],reportExponentialDoubleQl["actions"],reportExponentialSplitQl["actions"],reportSARSA["actions"]

# draw graphs of different algorithms in IGT
def drawGraphIGT(reportQl, reportDoubleQl, reportSplitQl,reportPositiveQl,reportNegativeQl,reportExponentialDoubleQl,reportExponentialSplitQl,reportSARSA,reward_from_A,reward_from_B,reward_from_C,reward_from_D,fig_name,draw_Exponential=False,draw_SARSA=False):
    plt.rcParams['figure.figsize'] = [20, 10]
    lw = 2
    
    yQL,yDQL,ySQL,yPQL,yNQL,yEDQL,yESQL,ySARSA = reportQl['percent'],reportDoubleQl['percent'],reportSplitQl['percent'],reportPositiveQl['percent'],reportNegativeQl['percent'],reportExponentialDoubleQl['percent'],reportExponentialSplitQl['percent'],reportSARSA['percent']
    yQL,yDQL,ySQL,yPQL,yNQL,yEDQL,yESQL,ySARSA = 100-yQL,100-yDQL,100-ySQL,100-yPQL,100-yNQL,100-yEDQL,100-yESQL,100-ySARSA
    
    aQL1,aDQL1,aSQL1,aPQL1,aNQL1,aEDQL1,aESQL1,aSARSA1 = reportQl["Q1(I)a"],reportDoubleQl["Q1(I)a"],reportSplitQl["Q1(I)a"],reportPositiveQl["Q1(I)a"],reportNegativeQl["Q1(I)a"],reportExponentialDoubleQl["Q1(I)a"],reportExponentialSplitQl["Q1(I)a"],reportSARSA["Q1(I)d"]
    bQL1,bDQL1,bSQL1,bPQL1,bNQL1,bEDQL1,bESQL1,bSARSA1 = reportQl["Q1(I)b"],reportDoubleQl["Q1(I)b"],reportSplitQl["Q1(I)b"],reportPositiveQl["Q1(I)b"],reportNegativeQl["Q1(I)b"],reportExponentialDoubleQl["Q1(I)b"],reportExponentialSplitQl["Q1(I)b"],reportSARSA["Q1(I)b"]
    cQL1,cDQL1,cSQL1,cPQL1,cNQL1,cEDQL1,cESQL1,cSARSA1 = reportQl["Q1(I)c"],reportDoubleQl["Q1(I)c"],reportSplitQl["Q1(I)c"],reportPositiveQl["Q1(I)c"],reportNegativeQl["Q1(I)c"],reportExponentialDoubleQl["Q1(I)c"],reportExponentialSplitQl["Q1(I)c"],reportSARSA["Q1(I)c"]
    dQL1,dDQL1,dSQL1,dPQL1,dNQL1,dEDQL1,dESQL1,dSARSA1 = reportQl["Q1(I)d"],reportDoubleQl["Q1(I)d"],reportSplitQl["Q1(I)d"],reportPositiveQl["Q1(I)d"],reportNegativeQl["Q1(I)d"],reportExponentialDoubleQl["Q1(I)d"],reportExponentialSplitQl["Q1(I)d"],reportSARSA["Q1(I)d"]
    aQL2,aDQL2,aSQL2,aPQL2,aNQL2,aEDQL2,aESQL2,aSARSA2 = reportQl["Q1(I)a"],reportDoubleQl["Q2(I)a"],reportSplitQl["Q2(I)a"],reportPositiveQl["Q2(I)a"],reportNegativeQl["Q2(I)a"],reportExponentialDoubleQl["Q2(I)a"],reportExponentialSplitQl["Q2(I)a"],reportSARSA["Q1(I)a"]
    bQL2,bDQL2,bSQL2,bPQL2,bNQL2,bEDQL2,bESQL2,bSARSA2 = reportQl["Q1(I)b"],reportDoubleQl["Q2(I)b"],reportSplitQl["Q2(I)b"],reportPositiveQl["Q2(I)b"],reportNegativeQl["Q2(I)b"],reportExponentialDoubleQl["Q2(I)b"],reportExponentialSplitQl["Q2(I)b"],reportSARSA["Q1(I)b"]
    cQL2,cDQL2,cSQL2,cPQL2,cNQL2,cEDQL2,cESQL2,cSARSA2 = reportQl["Q1(I)c"],reportDoubleQl["Q2(I)c"],reportSplitQl["Q2(I)c"],reportPositiveQl["Q2(I)c"],reportNegativeQl["Q2(I)c"],reportExponentialDoubleQl["Q2(I)c"],reportExponentialSplitQl["Q2(I)c"],reportSARSA["Q1(I)c"]
    dQL2,dDQL2,dSQL2,dPQL2,dNQL2,dEDQL2,dESQL2,dSARSA2 = reportQl["Q1(I)d"],reportDoubleQl["Q2(I)d"],reportSplitQl["Q2(I)d"],reportPositiveQl["Q2(I)d"],reportNegativeQl["Q2(I)d"],reportExponentialDoubleQl["Q2(I)d"],reportExponentialSplitQl["Q2(I)d"],reportSARSA["Q1(I)d"]
    rQL,rDQL,rSQL,rPQL,rNQL,rEDQL,rESQL,rSARSA = np.cumsum(reportQl["reward"],1),np.cumsum(reportDoubleQl["reward"],1),np.cumsum(reportSplitQl["reward"],1),np.cumsum(reportPositiveQl["reward"],1),np.cumsum(reportNegativeQl["reward"],1),np.cumsum(reportExponentialDoubleQl["reward"],1),np.cumsum(reportExponentialSplitQl["reward"],1),np.cumsum(reportSARSA["reward"],1)
  
    steps = np.arange(yQL.shape[1])+1

    fig = plt.figure()    
    fig.subplots_adjust(hspace=0.5)

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

    data = []
    for i in range(1000):
        a = np.sum(reward_from_A(i))
        b = np.sum(reward_from_B(i))
        c = np.sum(reward_from_C(i))
        d = np.sum(reward_from_D(i))
        data.append([a,b,c,d])
    data = pd.DataFrame(data, columns=['A','B','C','D'])
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

    l_ql, = axlist[0][1].plot(steps, np.mean(yQL,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[0][1].plot(steps, np.mean(yDQL,0), marker='', color='blue',linewidth=lw, label="DQL")
    l_sql, = axlist[0][1].plot(steps, np.mean(ySQL,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[0][1].plot(steps, np.mean(yPQL,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[0][1].plot(steps, np.mean(yNQL,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[0][1].plot(steps, np.mean(ySARSA,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[0][1].plot(steps, np.mean(yEDQL,0), marker='', color='purple',linewidth=lw, label="EDQL")
        l_esql, = axlist[0][1].plot(steps, np.mean(yESQL,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[0][1].set_xlabel('Episodes')
    axlist[0][1].set_ylim(0,100)
    axlist[0][1].set_ylabel('% choosing better decks')
    axlist[0][1].grid(True)

    l_ql, = axlist[0][2].plot(steps, np.mean(rQL,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[0][2].plot(steps, np.mean(rDQL,0), marker='', color='blue',linewidth=lw, label="DQL")
    l_sql, = axlist[0][2].plot(steps, np.mean(rSQL,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[0][2].plot(steps, np.mean(rPQL,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[0][2].plot(steps, np.mean(rNQL,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[0][2].plot(steps, np.mean(rSARSA,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[0][2].plot(steps, np.mean(rEDQL,0), marker='', color='purple',linewidth=lw, label="EDQL")
        l_esql, = axlist[0][2].plot(steps, np.mean(rESQL,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[0][2].set_xlabel('Episodes')
    axlist[0][2].set_ylabel('Cumulative episode rewards')
    axlist[0][2].grid(True)
    
    l_ql, = axlist[1][0].plot(steps, np.mean(aQL1,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[1][0].plot(steps, np.mean(aDQL1,0), marker='', color='blue',linewidth=lw, label="DQL")
    l_sql, = axlist[1][0].plot(steps, np.mean(aSQL1,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[1][0].plot(steps, np.mean(aPQL1,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[1][0].plot(steps, np.mean(aNQL1,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[1][0].plot(steps, np.mean(aSARSA1,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[1][0].plot(steps, np.mean(aEDQL1,0), marker='', color='purple',linewidth=lw, label="EDQL")
        l_esql, = axlist[1][0].plot(steps, np.mean(aESQL1,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[1][0].set_xlabel('Episodes')
    axlist[1][0].set_ylabel('Q1 for picking A')
    axlist[1][0].grid(True)
 
    l_ql, = axlist[1][1].plot(steps, np.mean(bQL1,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[1][1].plot(steps, np.mean(bDQL1,0), marker='', color='blue',linewidth=lw, label="DQL")
    l_sql, = axlist[1][1].plot(steps, np.mean(bSQL1,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[1][1].plot(steps, np.mean(bPQL1,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[1][1].plot(steps, np.mean(bNQL1,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[1][1].plot(steps, np.mean(bSARSA1,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[1][1].plot(steps, np.mean(bEDQL1,0), marker='', color='purple',linewidth=lw, label="EDQL")
        l_esql, = axlist[1][1].plot(steps, np.mean(bESQL1,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[1][1].set_xlabel('Episodes')
    axlist[1][1].set_ylabel('Q1 for picking B')
    axlist[1][1].grid(True)

    l_ql, = axlist[1][2].plot(steps, np.mean(cQL1,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[1][2].plot(steps, np.mean(cDQL1,0), marker='', color='blue',linewidth=lw, label="DQL")
    l_sql, = axlist[1][2].plot(steps, np.mean(cSQL1,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[1][2].plot(steps, np.mean(cPQL1,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[1][2].plot(steps, np.mean(cNQL1,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[1][2].plot(steps, np.mean(cSARSA1,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[1][2].plot(steps, np.mean(cEDQL1,0), marker='', color='purple',linewidth=lw, label="EDQL")
        l_esql, = axlist[1][2].plot(steps, np.mean(cESQL1,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[1][2].set_xlabel('Episodes')
    axlist[1][2].set_ylabel('Q1 for picking C')
    axlist[1][2].grid(True)
 
    l_ql, = axlist[1][3].plot(steps, np.mean(dQL1,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[1][3].plot(steps, np.mean(dDQL1,0), marker='', color='blue',linewidth=lw, label="DQL")
    l_sql, = axlist[1][3].plot(steps, np.mean(dSQL1,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[1][3].plot(steps, np.mean(dPQL1,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[1][3].plot(steps, np.mean(dNQL1,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[1][3].plot(steps, np.mean(dSARSA1,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[1][3].plot(steps, np.mean(dEDQL1,0), marker='', color='purple',linewidth=lw, label="EDQL")
        l_esql, = axlist[1][3].plot(steps, np.mean(dESQL1,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[1][3].set_xlabel('Episodes')
    axlist[1][3].set_ylabel('Q1 for picking D')
    axlist[1][3].grid(True)
 
     
    l_ql, = axlist[2][0].plot(steps, np.mean(aQL2,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[2][0].plot(steps, np.mean(aDQL2,0), marker='', color='blue', linewidth=lw, label="DQL")
    l_sql, = axlist[2][0].plot(steps, np.mean(aSQL2,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[2][0].plot(steps, np.mean(aPQL2,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[2][0].plot(steps, np.mean(aNQL2,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[2][0].plot(steps, np.mean(aSARSA2,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[2][0].plot(steps, np.mean(aEDQL2,0), marker='', color='purple', linewidth=lw, label="EDQL")
        l_esql, = axlist[2][0].plot(steps, np.mean(aESQL2,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[2][0].set_xlabel('Episodes')
    axlist[2][0].set_ylabel('Q2 for picking A')
    axlist[2][0].grid(True)
 
    l_ql, = axlist[2][1].plot(steps, np.mean(bQL2,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[2][1].plot(steps, np.mean(bDQL2,0), marker='', color='blue', linewidth=lw, label="DQL")
    l_sql, = axlist[2][1].plot(steps, np.mean(bSQL2,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[2][1].plot(steps, np.mean(bPQL2,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[2][1].plot(steps, np.mean(bNQL2,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[2][1].plot(steps, np.mean(bSARSA2,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[2][1].plot(steps, np.mean(bEDQL2,0), marker='', color='purple', linewidth=lw, label="EDQL")
        l_esql, = axlist[2][1].plot(steps, np.mean(bESQL2,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[2][1].set_xlabel('Episodes')
    axlist[2][1].set_ylabel('Q2 for picking B')
    axlist[2][1].grid(True)

    l_ql, = axlist[2][2].plot(steps, np.mean(cQL2,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[2][2].plot(steps, np.mean(cDQL2,0), marker='', color='blue', linewidth=lw, label="DQL")
    l_sql, = axlist[2][2].plot(steps, np.mean(cSQL2,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[2][2].plot(steps, np.mean(cPQL2,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[2][2].plot(steps, np.mean(cNQL2,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[2][2].plot(steps, np.mean(cSARSA2,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[2][2].plot(steps, np.mean(cEDQL2,0), marker='', color='purple', linewidth=lw, label="EDQL")
        l_esql, = axlist[2][2].plot(steps, np.mean(cESQL2,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[2][2].set_xlabel('Episodes')
    axlist[2][2].set_ylabel('Q2 for picking C')
    axlist[2][2].grid(True)
 
    l_ql, = axlist[2][3].plot(steps, np.mean(dQL2,0) , marker='', color='black', linewidth=lw, label="QL")
    l_dql, = axlist[2][3].plot(steps, np.mean(dDQL2,0), marker='', color='blue', linewidth=lw, label="DQL")
    l_sql, = axlist[2][3].plot(steps, np.mean(dSQL2,0), marker='', color='orange',linewidth=lw, label="SQL")
    l_pql, = axlist[2][3].plot(steps, np.mean(dPQL2,0), marker='', color='red',linewidth=lw, label="PQL")
    l_nql, = axlist[2][3].plot(steps, np.mean(dNQL2,0), marker='', color='green',linewidth=lw, label="NQL")
    if draw_SARSA:
        l_sarsa, = axlist[2][3].plot(steps, np.mean(dSARSA2,0), marker='', color='gold',linewidth=lw, label="SARSA")
    if draw_Exponential:
        l_edql, = axlist[2][3].plot(steps, np.mean(dEDQL2,0), marker='', color='purple', linewidth=lw, label="EDQL")
        l_esql, = axlist[2][3].plot(steps, np.mean(dESQL2,0), marker='', color='brown',linewidth=lw, label="ESQL")
    axlist[2][3].set_xlabel('Episodes')
    axlist[2][3].set_ylabel('Q2 for picking D')
    axlist[2][3].legend()
    axlist[2][3].grid(True)   

    alpha_plot = 0.2
    
    l_ql= axlist[0][1].fill_between(steps, np.mean(yQL,0)-np.std(yQL,0)/math.sqrt(cntExperiments),np.mean(yQL,0)+np.std(yQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_dql= axlist[0][1].fill_between(steps, np.mean(yDQL,0)-np.std(yDQL,0)/math.sqrt(cntExperiments),np.mean(yDQL,0)+np.std(yDQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_sql= axlist[0][1].fill_between(steps, np.mean(ySQL,0)-np.std(ySQL,0)/math.sqrt(cntExperiments),np.mean(ySQL,0)+np.std(ySQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_pql= axlist[0][1].fill_between(steps, np.mean(yPQL,0)-np.std(yPQL,0)/math.sqrt(cntExperiments),np.mean(yPQL,0)+np.std(yPQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_nql= axlist[0][1].fill_between(steps, np.mean(yNQL,0)-np.std(yNQL,0)/math.sqrt(cntExperiments),np.mean(yNQL,0)+np.std(yNQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    if draw_SARSA:
        l_sarsa= axlist[0][1].fill_between(steps, np.mean(ySARSA,0)-np.std(ySARSA,0)/math.sqrt(cntExperiments),np.mean(ySARSA,0)+np.std(ySARSA,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    if draw_Exponential:
        l_edql= axlist[0][1].fill_between(steps, np.mean(yEDQL,0)-np.std(yEDQL,0)/math.sqrt(cntExperiments),np.mean(yEDQL,0)+np.std(yEDQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_esql= axlist[0][1].fill_between(steps, np.mean(yESQL,0)-np.std(yESQL,0)/math.sqrt(cntExperiments),np.mean(yESQL,0)+np.std(yESQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

    l_ql= axlist[0][2].fill_between(steps, np.mean(rQL,0)-np.std(rQL,0)/math.sqrt(cntExperiments),np.mean(rQL,0)+np.std(rQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_dql= axlist[0][2].fill_between(steps, np.mean(rDQL,0)-np.std(rDQL,0)/math.sqrt(cntExperiments),np.mean(rDQL,0)+np.std(rDQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_sql= axlist[0][2].fill_between(steps, np.mean(rSQL,0)-np.std(rSQL,0)/math.sqrt(cntExperiments),np.mean(rSQL,0)+np.std(rSQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_pql= axlist[0][2].fill_between(steps, np.mean(rPQL,0)-np.std(rPQL,0)/math.sqrt(cntExperiments),np.mean(rPQL,0)+np.std(rPQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_nql= axlist[0][2].fill_between(steps, np.mean(rNQL,0)-np.std(rNQL,0)/math.sqrt(cntExperiments),np.mean(rNQL,0)+np.std(rNQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    if draw_SARSA:
        l_sarsa= axlist[0][2].fill_between(steps, np.mean(rSARSA,0)-np.std(rSARSA,0)/math.sqrt(cntExperiments),np.mean(rSARSA,0)+np.std(rSARSA,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    if draw_Exponential:
        l_edql= axlist[0][2].fill_between(steps, np.mean(rEDQL,0)-np.std(rEDQL,0)/math.sqrt(cntExperiments),np.mean(rEDQL,0)+np.std(rEDQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_esql= axlist[0][2].fill_between(steps, np.mean(rESQL,0)-np.std(rESQL,0)/math.sqrt(cntExperiments),np.mean(rESQL,0)+np.std(rESQL,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
    
    l_ql= axlist[1][0].fill_between(steps, np.mean(aQL1,0)-np.std(aQL1,0)/math.sqrt(cntExperiments),np.mean(aQL1,0)+np.std(aQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_dql= axlist[1][0].fill_between(steps, np.mean(aDQL1,0)-np.std(aDQL1,0)/math.sqrt(cntExperiments),np.mean(aDQL1,0)+np.std(aDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_sql= axlist[1][0].fill_between(steps, np.mean(aSQL1,0)-np.std(aSQL1,0)/math.sqrt(cntExperiments),np.mean(aSQL1,0)+np.std(aSQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_pql= axlist[1][0].fill_between(steps, np.mean(aPQL1,0)-np.std(aPQL1,0)/math.sqrt(cntExperiments),np.mean(aPQL1,0)+np.std(aPQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_nql= axlist[1][0].fill_between(steps, np.mean(aNQL1,0)-np.std(aNQL1,0)/math.sqrt(cntExperiments),np.mean(aNQL1,0)+np.std(aNQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    if draw_SARSA:
        l_sarsa= axlist[1][0].fill_between(steps, np.mean(aSARSA1,0)-np.std(aSARSA1,0)/math.sqrt(cntExperiments),np.mean(aSARSA1,0)+np.std(aSARSA1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    if draw_Exponential:
        l_edql= axlist[1][0].fill_between(steps, np.mean(aEDQL1,0)-np.std(aEDQL1,0)/math.sqrt(cntExperiments),np.mean(aEDQL1,0)+np.std(aEDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_esql= axlist[1][0].fill_between(steps, np.mean(aESQL1,0)-np.std(aESQL1,0)/math.sqrt(cntExperiments),np.mean(aESQL1,0)+np.std(aESQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
 
    l_ql= axlist[1][1].fill_between(steps, np.mean(bQL1,0)-np.std(bQL1,0)/math.sqrt(cntExperiments),np.mean(bQL1,0)+np.std(bQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_dql= axlist[1][1].fill_between(steps, np.mean(bDQL1,0)-np.std(bDQL1,0)/math.sqrt(cntExperiments),np.mean(bDQL1,0)+np.std(bDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_sql= axlist[1][1].fill_between(steps, np.mean(bSQL1,0)-np.std(bSQL1,0)/math.sqrt(cntExperiments),np.mean(bSQL1,0)+np.std(bSQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_pql= axlist[1][1].fill_between(steps, np.mean(bPQL1,0)-np.std(bPQL1,0)/math.sqrt(cntExperiments),np.mean(bPQL1,0)+np.std(bPQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_nql= axlist[1][1].fill_between(steps, np.mean(bNQL1,0)-np.std(bNQL1,0)/math.sqrt(cntExperiments),np.mean(bNQL1,0)+np.std(bNQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    if draw_SARSA:
        l_sarsa= axlist[1][1].fill_between(steps, np.mean(bSARSA1,0)-np.std(bSARSA1,0)/math.sqrt(cntExperiments),np.mean(bSARSA1,0)+np.std(bSARSA1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    if draw_Exponential:
        l_edql= axlist[1][1].fill_between(steps, np.mean(bEDQL1,0)-np.std(bEDQL1,0)/math.sqrt(cntExperiments),np.mean(bEDQL1,0)+np.std(bEDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_esql= axlist[1][1].fill_between(steps, np.mean(bESQL1,0)-np.std(bESQL1,0)/math.sqrt(cntExperiments),np.mean(bESQL1,0)+np.std(bESQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

    l_ql= axlist[1][2].fill_between(steps, np.mean(cQL1,0)-np.std(cQL1,0)/math.sqrt(cntExperiments),np.mean(cQL1,0)+np.std(cQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_dql= axlist[1][2].fill_between(steps, np.mean(cDQL1,0)-np.std(cDQL1,0)/math.sqrt(cntExperiments),np.mean(cDQL1,0)+np.std(cDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_sql= axlist[1][2].fill_between(steps, np.mean(cSQL1,0)-np.std(cSQL1,0)/math.sqrt(cntExperiments),np.mean(cSQL1,0)+np.std(cSQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_pql= axlist[1][2].fill_between(steps, np.mean(cPQL1,0)-np.std(cPQL1,0)/math.sqrt(cntExperiments),np.mean(cPQL1,0)+np.std(cPQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_nql= axlist[1][2].fill_between(steps, np.mean(cNQL1,0)-np.std(cNQL1,0)/math.sqrt(cntExperiments),np.mean(cNQL1,0)+np.std(cNQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    if draw_SARSA:
        l_sarsa= axlist[1][2].fill_between(steps, np.mean(cSARSA1,0)-np.std(cSARSA1,0)/math.sqrt(cntExperiments),np.mean(cSARSA1,0)+np.std(cSARSA1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    if draw_Exponential:
        l_edql= axlist[1][2].fill_between(steps, np.mean(cEDQL1,0)-np.std(cEDQL1,0)/math.sqrt(cntExperiments),np.mean(cEDQL1,0)+np.std(cEDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_esql= axlist[1][2].fill_between(steps, np.mean(cESQL1,0)-np.std(cESQL1,0)/math.sqrt(cntExperiments),np.mean(cESQL1,0)+np.std(cESQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
 
    l_ql= axlist[1][3].fill_between(steps, np.mean(dQL1,0)-np.std(dQL1,0)/math.sqrt(cntExperiments),np.mean(dQL1,0)+np.std(dQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_dql= axlist[1][3].fill_between(steps, np.mean(dDQL1,0)-np.std(dDQL1,0)/math.sqrt(cntExperiments),np.mean(dDQL1,0)+np.std(dDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_sql= axlist[1][3].fill_between(steps, np.mean(dSQL1,0)-np.std(dSQL1,0)/math.sqrt(cntExperiments),np.mean(dSQL1,0)+np.std(dSQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_pql= axlist[1][3].fill_between(steps, np.mean(dPQL1,0)-np.std(dPQL1,0)/math.sqrt(cntExperiments),np.mean(dPQL1,0)+np.std(dPQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_nql= axlist[1][3].fill_between(steps, np.mean(dNQL1,0)-np.std(dNQL1,0)/math.sqrt(cntExperiments),np.mean(dNQL1,0)+np.std(dNQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    if draw_SARSA:
        l_sarsa= axlist[1][3].fill_between(steps, np.mean(dSARSA1,0)-np.std(dSARSA1,0)/math.sqrt(cntExperiments),np.mean(dSARSA1,0)+np.std(dSARSA1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    if draw_Exponential:
        l_edql= axlist[1][3].fill_between(steps, np.mean(dEDQL1,0)-np.std(dEDQL1,0)/math.sqrt(cntExperiments),np.mean(dEDQL1,0)+np.std(dEDQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_esql= axlist[1][3].fill_between(steps, np.mean(dESQL1,0)-np.std(dESQL1,0)/math.sqrt(cntExperiments),np.mean(dESQL1,0)+np.std(dESQL1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
     
    l_ql= axlist[2][0].fill_between(steps, np.mean(aQL2,0)-np.std(aQL2,0)/math.sqrt(cntExperiments),np.mean(aQL2,0)+np.std(aQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_dql= axlist[2][0].fill_between(steps, np.mean(aDQL2,0)-np.std(aDQL2,0)/math.sqrt(cntExperiments),np.mean(aDQL2,0)+np.std(aDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_sql= axlist[2][0].fill_between(steps, np.mean(aSQL2,0)-np.std(aSQL2,0)/math.sqrt(cntExperiments),np.mean(aSQL2,0)+np.std(aSQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_pql= axlist[2][0].fill_between(steps, np.mean(aPQL2,0)-np.std(aPQL2,0)/math.sqrt(cntExperiments),np.mean(aPQL2,0)+np.std(aPQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_nql= axlist[2][0].fill_between(steps, np.mean(aNQL2,0)-np.std(aNQL2,0)/math.sqrt(cntExperiments),np.mean(aNQL2,0)+np.std(aNQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    if draw_SARSA:
        l_sarsa= axlist[2][0].fill_between(steps, np.mean(aSARSA2,0)-np.std(aSARSA2,0)/math.sqrt(cntExperiments),np.mean(aSARSA2,0)+np.std(aSARSA2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    if draw_Exponential:
        l_edql= axlist[2][0].fill_between(steps, np.mean(aEDQL2,0)-np.std(aEDQL2,0)/math.sqrt(cntExperiments),np.mean(aEDQL2,0)+np.std(aEDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_esql= axlist[2][0].fill_between(steps, np.mean(aESQL2,0)-np.std(aESQL2,0)/math.sqrt(cntExperiments),np.mean(aESQL2,0)+np.std(aESQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
 
    l_ql= axlist[2][1].fill_between(steps, np.mean(bQL2,0)-np.std(bQL2,0)/math.sqrt(cntExperiments),np.mean(bQL2,0)+np.std(bQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_dql= axlist[2][1].fill_between(steps, np.mean(bDQL2,0)-np.std(bDQL2,0)/math.sqrt(cntExperiments),np.mean(bDQL2,0)+np.std(bDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_sql= axlist[2][1].fill_between(steps, np.mean(bSQL2,0)-np.std(bSQL2,0)/math.sqrt(cntExperiments),np.mean(bSQL2,0)+np.std(bSQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_pql= axlist[2][1].fill_between(steps, np.mean(bPQL2,0)-np.std(bPQL2,0)/math.sqrt(cntExperiments),np.mean(bPQL2,0)+np.std(bPQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_nql= axlist[2][1].fill_between(steps, np.mean(bNQL2,0)-np.std(bNQL2,0)/math.sqrt(cntExperiments),np.mean(bNQL2,0)+np.std(bNQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    if draw_SARSA:
        l_sarsa= axlist[2][1].fill_between(steps, np.mean(bSARSA2,0)-np.std(bSARSA2,0)/math.sqrt(cntExperiments),np.mean(bSARSA2,0)+np.std(bSARSA2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    if draw_Exponential:
        l_edql= axlist[2][1].fill_between(steps, np.mean(bEDQL2,0)-np.std(bEDQL2,0)/math.sqrt(cntExperiments),np.mean(bEDQL2,0)+np.std(bEDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_esql= axlist[2][1].fill_between(steps, np.mean(bESQL2,0)-np.std(bESQL2,0)/math.sqrt(cntExperiments),np.mean(bESQL2,0)+np.std(bESQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

    l_ql= axlist[2][2].fill_between(steps, np.mean(cQL2,0)-np.std(cQL2,0)/math.sqrt(cntExperiments),np.mean(cQL2,0)+np.std(cQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_dql= axlist[2][2].fill_between(steps, np.mean(cDQL2,0)-np.std(cDQL2,0)/math.sqrt(cntExperiments),np.mean(cDQL2,0)+np.std(cDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_sql= axlist[2][2].fill_between(steps, np.mean(cSQL2,0)-np.std(cSQL2,0)/math.sqrt(cntExperiments),np.mean(cSQL2,0)+np.std(cSQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_pql= axlist[2][2].fill_between(steps, np.mean(cPQL2,0)-np.std(cPQL2,0)/math.sqrt(cntExperiments),np.mean(cPQL2,0)+np.std(cPQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_nql= axlist[2][2].fill_between(steps, np.mean(cNQL2,0)-np.std(cNQL2,0)/math.sqrt(cntExperiments),np.mean(cNQL2,0)-np.std(cNQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    if draw_SARSA:
        l_sarsa= axlist[2][2].fill_between(steps, np.mean(cSARSA2,0)-np.std(cSARSA2,0)/math.sqrt(cntExperiments),np.mean(cSARSA2,0)+np.std(cSARSA2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    if draw_Exponential:
        l_edql= axlist[2][2].fill_between(steps, np.mean(cEDQL2,0)-np.std(cEDQL2,0)/math.sqrt(cntExperiments),np.mean(cEDQL2,0)+np.std(cEDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_esql= axlist[2][2].fill_between(steps, np.mean(cESQL2,0)-np.std(cESQL2,0)/math.sqrt(cntExperiments),np.mean(cESQL2,0)+np.std(cESQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
 
    l_ql= axlist[2][3].fill_between(steps, np.mean(dQL2,0)-np.std(dQL2,0)/math.sqrt(cntExperiments),np.mean(dQL2,0)+np.std(dQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_dql= axlist[2][3].fill_between(steps, np.mean(dDQL2,0)-np.std(dDQL2,0)/math.sqrt(cntExperiments),np.mean(dDQL2,0)+np.std(dDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_sql= axlist[2][3].fill_between(steps, np.mean(dSQL2,0)-np.std(dSQL2,0)/math.sqrt(cntExperiments),np.mean(dSQL2,0)+np.std(dSQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_pql= axlist[2][3].fill_between(steps, np.mean(dPQL2,0)-np.std(dPQL2,0)/math.sqrt(cntExperiments),np.mean(dPQL2,0)+np.std(dPQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_nql= axlist[2][3].fill_between(steps, np.mean(dNQL2,0)-np.std(dNQL2,0)/math.sqrt(cntExperiments),np.mean(dNQL2,0)+np.std(dNQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    if draw_SARSA:
        l_sarsa= axlist[2][3].fill_between(steps, np.mean(dSARSA2,0)-np.std(dSARSA2,0)/math.sqrt(cntExperiments),np.mean(dSARSA2,0)+np.std(dSARSA2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    if draw_Exponential:
        l_edql= axlist[2][3].fill_between(steps, np.mean(dEDQL2,0)-np.std(dEDQL2,0)/math.sqrt(cntExperiments),np.mean(dEDQL2,0)+np.std(dEDQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_esql= axlist[2][3].fill_between(steps, np.mean(dESQL2,0)-np.std(dESQL2,0)/math.sqrt(cntExperiments),np.mean(dESQL2,0)+np.std(dESQL2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

    fig.tight_layout() 
    fig.savefig(fig_name)
    
    return fig, np.mean(rQL,0)[-1],np.mean(rDQL,0)[-1],np.mean(rSQL,0)[-1],np.mean(rPQL,0)[-1],np.mean(rNQL,0)[-1],np.mean(rEDQL,0)[-1],np.mean(rESQL,0)[-1],np.mean(rSARSA,0)[-1],reportQl["reward"],reportDoubleQl["reward"],reportSplitQl["reward"],reportPositiveQl["reward"],reportNegativeQl["reward"],reportExponentialDoubleQl["reward"],reportExponentialSplitQl["reward"],reportSARSA["reward"],reportQl["actions"],reportDoubleQl["actions"],reportSplitQl["actions"],reportPositiveQl["actions"],reportNegativeQl["actions"],reportExponentialDoubleQl["actions"],reportExponentialSplitQl["actions"],reportSARSA["actions"]


# draw graphs of different RL algorithms
def drawGraph_mental(reportADD, reportADHD, reportAD,reportCP,reportbvFTD,reportPD,reportM,reward_from_B,reward_from_C,label_B,label_C,fig_name):
    plt.rcParams['figure.figsize'] = [20, 10]
    lw = 2
    
    yADD,yADHD,yAD,yCP,ybvFTD,yPD,yM = reportADD['percent'],reportADHD['percent'],reportAD['percent'],reportCP['percent'],reportbvFTD['percent'],reportPD['percent'],reportM['percent']
    yADD,yADHD,yAD,yCP,ybvFTD,yPD,yM = 100-yADD,100-yADHD,100-yAD,100-yCP,100-ybvFTD,100-yPD,100-yM
    lADD1,lADHD1,lAD1,lCP1,lbvFTD1,lPD1,lM1 = reportADD["Q1(A)l"],reportADHD["Q1(A)l"],reportAD["Q1(A)l"],reportCP["Q1(A)l"],reportbvFTD["Q1(A)l"],reportPD["Q1(A)l"],reportM["Q1(A)l"]
    rADD1,rADHD1,rAD1,rCP1,rbvFTD1,rPD1,rM1 = reportADD["Q1(A)r"],reportADHD["Q1(A)r"],reportAD["Q1(A)r"],reportCP["Q1(A)r"],reportbvFTD["Q1(A)r"],reportPD["Q1(A)r"],reportM["Q1(A)r"]
    lADD2,lADHD2,lAD2,lCP2,lbvFTD2,lPD2,lM2 = reportADD["Q2(A)l"],reportADHD["Q2(A)l"],reportAD["Q2(A)l"],reportCP["Q2(A)l"],reportbvFTD["Q2(A)l"],reportPD["Q2(A)l"],reportM["Q2(A)l"]
    rADD2,rADHD2,rAD2,rCP2,rbvFTD2,rPD2,rM2 = reportADD["Q2(A)r"],reportADHD["Q2(A)r"],reportAD["Q2(A)r"],reportCP["Q2(A)r"],reportbvFTD["Q2(A)r"],reportPD["Q2(A)r"],reportM["Q2(A)r"]
    cADD,cADHD,cAD,cCP,cbvFTD,cPD,cM = np.cumsum(reportADD["reward"],1),np.cumsum(reportADHD["reward"],1),np.cumsum(reportAD["reward"],1),np.cumsum(reportCP["reward"],1),np.cumsum(reportbvFTD["reward"],1),np.cumsum(reportPD["reward"],1),np.cumsum(reportM["reward"],1)
    
    steps = np.arange(yADD.shape[1])+1
      
    fig = plt.figure()    
    fig.subplots_adjust(hspace=0.5)

    ax1 = plt.subplot2grid((2, 12), (0, 0), colspan=4)
    ax2 = plt.subplot2grid((2, 12), (0, 4), colspan=4)
    ax3 = plt.subplot2grid((2, 12), (0, 8), colspan=4)
    ax4 = plt.subplot2grid((2, 12), (1, 0), colspan=3)
    ax5 = plt.subplot2grid((2, 12), (1, 3), colspan=3)
    ax6 = plt.subplot2grid((2, 12), (1, 6), colspan=3)
    ax7 = plt.subplot2grid((2, 12), (1, 9), colspan=3)
    axlist=[[ax1,ax2,ax3],[ax4,ax5,ax6,ax7]]
    
    data = []
    for i in range(5000):
        data.append([reward_from_B(),reward_from_C()])
    data = pd.DataFrame(data, columns=[label_B, label_C])
    sns.kdeplot(data[label_B],shade=True,ax=axlist[0][0])
    sns.kdeplot(data[label_C],shade=True,ax=axlist[0][0])
    axlist[0][0].axvline(x=np.mean(data[label_B]), color='blue', linestyle='--')
    axlist[0][0].axvline(x=np.mean(data[label_C]), color='orange', linestyle='--')
    axlist[0][0].set_xlabel('reward')
    axlist[0][0].set_ylabel('Reward distributions for action left vs. right')

    l_add, = axlist[0][1].plot(steps, np.mean(yADD,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[0][1].plot(steps, np.mean(yADHD,0), marker='', color='blue',linewidth=lw, label="ADHD")
    l_ad, = axlist[0][1].plot(steps, np.mean(yAD,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[0][1].plot(steps, np.mean(yCP,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[0][1].plot(steps, np.mean(ybvFTD,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[0][1].plot(steps, np.mean(yPD,0), marker='', color='purple',linewidth=lw, label="PD")
    l_m, = axlist[0][1].plot(steps, np.mean(yM,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[0][1].set_xlabel('Episodes')
    axlist[0][1].set_ylabel('% choosing better action')
    axlist[0][1].set_ylim(0,100)
    axlist[0][1].grid(True)

    l_add, = axlist[0][2].plot(steps, np.mean(cADD,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[0][2].plot(steps, np.mean(cADHD,0), marker='', color='blue',linewidth=lw, label="ADHD")
    l_ad, = axlist[0][2].plot(steps, np.mean(cAD,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[0][2].plot(steps, np.mean(cCP,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[0][2].plot(steps, np.mean(cbvFTD,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[0][2].plot(steps, np.mean(cPD,0), marker='', color='purple',linewidth=lw, label="PD")
    l_m, = axlist[0][2].plot(steps, np.mean(cM,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[0][2].set_xlabel('Episodes')
    axlist[0][2].set_ylabel('Cumulative episode rewards')
    axlist[0][2].grid(True)
    
    l_add, = axlist[1][0].plot(steps, np.mean(lADD1,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[1][0].plot(steps, np.mean(lADHD1,0), marker='', color='blue',linewidth=lw, label="ADHD")
    l_ad, = axlist[1][0].plot(steps, np.mean(lAD1,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[1][0].plot(steps, np.mean(lCP1,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[1][0].plot(steps, np.mean(lbvFTD1,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[1][0].plot(steps, np.mean(lPD1,0), marker='', color='purple',linewidth=lw, label="PD")
    l_m, = axlist[1][0].plot(steps, np.mean(lM1,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[1][0].set_xlabel('Episodes')
    axlist[1][0].set_ylabel('Q1 for action left at state A')
    axlist[1][0].grid(True)

    l_add, = axlist[1][1].plot(steps, np.mean(rADD1,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[1][1].plot(steps, np.mean(rADHD1,0), marker='', color='blue',linewidth=lw, label="ADHD")
    l_ad, = axlist[1][1].plot(steps, np.mean(rAD1,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[1][1].plot(steps, np.mean(rCP1,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[1][1].plot(steps, np.mean(rbvFTD1,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[1][1].plot(steps, np.mean(rPD1,0), marker='', color='purple',linewidth=lw, label="PD")
    l_m, = axlist[1][1].plot(steps, np.mean(rM1,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[1][1].set_xlabel('Episodes')
    axlist[1][1].set_ylabel('Q1 for action right at state A')
#     axlist[1][1].legend()
    axlist[1][1].grid(True)

    l_add, = axlist[1][2].plot(steps, np.mean(lADD2,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[1][2].plot(steps, np.mean(lADHD2,0), marker='', color='blue', linewidth=lw, label="ADHD")
    l_ad, = axlist[1][2].plot(steps, np.mean(lAD2,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[1][2].plot(steps, np.mean(lCP2,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[1][2].plot(steps, np.mean(lbvFTD2,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[1][2].plot(steps, np.mean(lPD2,0), marker='', color='purple', linewidth=lw, label="PD")
    l_m, = axlist[1][2].plot(steps, np.mean(lM2,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[1][2].set_xlabel('Episodes')
    axlist[1][2].set_ylabel('Q2 for action left at state A')
    axlist[1][2].grid(True)

    l_add, = axlist[1][3].plot(steps, np.mean(rADD2,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[1][3].plot(steps, np.mean(rADHD2,0), marker='', color='blue', linewidth=lw, label="ADHD")
    l_ad, = axlist[1][3].plot(steps, np.mean(rAD2,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[1][3].plot(steps, np.mean(rCP2,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[1][3].plot(steps, np.mean(rbvFTD2,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[1][3].plot(steps, np.mean(rPD2,0), marker='', color='purple', linewidth=lw, label="PD")
    l_m, = axlist[1][3].plot(steps, np.mean(rM2,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[1][3].set_xlabel('Episodes')
    axlist[1][3].set_ylabel('Q2 for action right at state A')
    axlist[1][3].legend()
    axlist[1][3].grid(True)
    
    alpha_plot = 0.2
    
    l_add  = axlist[0][1].fill_between(steps, np.mean(yADD,0)-np.std(yADD,0)/math.sqrt(cntExperiments),np.mean(yADD,0)+np.std(yADD,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_adhd = axlist[0][1].fill_between(steps, np.mean(yADHD,0)-np.std(yADHD,0)/math.sqrt(cntExperiments),np.mean(yADHD,0)+np.std(yADHD,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_ad = axlist[0][1].fill_between(steps, np.mean(yAD,0)-np.std(yAD,0)/math.sqrt(cntExperiments),np.mean(yAD,0)+np.std(yAD,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_cp = axlist[0][1].fill_between(steps, np.mean(yCP,0)-np.std(yCP,0)/math.sqrt(cntExperiments),np.mean(yCP,0)+np.std(yCP,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_bvftd = axlist[0][1].fill_between(steps, np.mean(ybvFTD,0)-np.std(ybvFTD,0)/math.sqrt(cntExperiments),np.mean(ybvFTD,0)+np.std(ybvFTD,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    l_pd = axlist[0][1].fill_between(steps, np.mean(yPD,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
    l_m = axlist[0][1].fill_between(steps, np.mean(yM,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')

    l_add = axlist[0][2].fill_between(steps, np.mean(cADD,0)-np.std(cADD,0)/math.sqrt(cntExperiments),np.mean(cADD,0)+np.std(cADD,0) /math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_adhd = axlist[0][2].fill_between(steps, np.mean(cADHD,0)-np.std(cADHD,0)/math.sqrt(cntExperiments),np.mean(cADHD,0)+np.std(cADHD,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_ad = axlist[0][2].fill_between(steps, np.mean(cAD,0)-np.std(cAD,0)/math.sqrt(cntExperiments),np.mean(cAD,0)+np.std(cAD,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_cp = axlist[0][2].fill_between(steps, np.mean(cCP,0)-np.std(cCP,0)/math.sqrt(cntExperiments),np.mean(cCP,0)+np.std(cCP,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_bvftd = axlist[0][2].fill_between(steps, np.mean(cbvFTD,0)-np.std(cbvFTD,0)/math.sqrt(cntExperiments),np.mean(cbvFTD,0)+np.std(cbvFTD,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    l_pd = axlist[0][2].fill_between(steps, np.mean(cPD,0)-np.std(cPD,0)/math.sqrt(cntExperiments),np.mean(cPD,0)+np.std(cPD,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
    l_m = axlist[0][2].fill_between(steps, np.mean(cM,0)-np.std(cM,0)/math.sqrt(cntExperiments),np.mean(cM,0)+np.std(cM,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')
    
    l_add = axlist[1][0].fill_between(steps, np.mean(lADD1,0)-np.std(lADD1,0)/math.sqrt(cntExperiments),np.mean(lADD1,0)-np.std(lADD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_adhd = axlist[1][0].fill_between(steps, np.mean(lADHD1,0)-np.std(lADHD1,0)/math.sqrt(cntExperiments),np.mean(lADHD1,0)+np.std(lADHD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_ad = axlist[1][0].fill_between(steps, np.mean(lAD1,0)-np.std(lAD1,0)/math.sqrt(cntExperiments),np.mean(lAD1,0)+np.std(lAD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_cp = axlist[1][0].fill_between(steps, np.mean(lCP1,0)-np.std(lCP1,0)/math.sqrt(cntExperiments),np.mean(lCP1,0)+np.std(lCP1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_bvftd = axlist[1][0].fill_between(steps, np.mean(lbvFTD1,0)-np.std(lbvFTD1,0)/math.sqrt(cntExperiments),np.mean(lbvFTD1,0)+np.std(lbvFTD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    l_pd = axlist[1][0].fill_between(steps, np.mean(lPD1,0)-np.std(lPD1,0)/math.sqrt(cntExperiments),np.mean(lPD1,0)+np.std(lPD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
    l_m = axlist[1][0].fill_between(steps, np.mean(lM1,0)-np.std(lM1,0)/math.sqrt(cntExperiments),np.mean(lM1,0)+np.std(lM1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')

    l_add = axlist[1][1].fill_between(steps, np.mean(rADD1,0)-np.std(rADD1,0)/math.sqrt(cntExperiments),np.mean(rADD1,0)+np.std(rADD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_adhd = axlist[1][1].fill_between(steps, np.mean(rADHD1,0)-np.std(rADHD1,0)/math.sqrt(cntExperiments),np.mean(rADHD1,0)+np.std(rADHD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_ad = axlist[1][1].fill_between(steps, np.mean(rAD1,0)-np.std(rAD1,0)/math.sqrt(cntExperiments),np.mean(rAD1,0)+np.std(rAD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_cp = axlist[1][1].fill_between(steps, np.mean(rCP1,0)-np.std(rCP1,0)/math.sqrt(cntExperiments),np.mean(rCP1,0)+np.std(rCP1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_bvftd = axlist[1][1].fill_between(steps, np.mean(rbvFTD1,0)-np.std(rbvFTD1,0)/math.sqrt(cntExperiments),np.mean(rbvFTD1,0)+np.std(rbvFTD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    l_pd = axlist[1][1].fill_between(steps, np.mean(rPD1,0)-np.std(rPD1,0)/math.sqrt(cntExperiments),np.mean(rPD1,0)+np.std(rPD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
    l_m = axlist[1][1].fill_between(steps, np.mean(rM1,0)-np.std(rM1,0)/math.sqrt(cntExperiments),np.mean(rM1,0)+np.std(rM1,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')


    l_add = axlist[1][2].fill_between(steps, np.mean(lADD2,0)-np.std(lADD2,0)/math.sqrt(cntExperiments),np.mean(lADD2,0)+np.std(lADD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_adhd = axlist[1][2].fill_between(steps, np.mean(lADHD2,0)-np.std(lADHD2,0)/math.sqrt(cntExperiments),np.mean(lADHD2,0)+np.std(lADHD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_ad = axlist[1][2].fill_between(steps, np.mean(lAD2,0)-np.std(lAD2,0)/math.sqrt(cntExperiments),np.mean(lAD2,0)+np.std(lAD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_cp = axlist[1][2].fill_between(steps, np.mean(lCP2,0)-np.std(lCP2,0)/math.sqrt(cntExperiments),np.mean(lCP2,0)+np.std(lCP2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_bvftd = axlist[1][2].fill_between(steps, np.mean(lbvFTD2,0)-np.std(lbvFTD2,0)/math.sqrt(cntExperiments),np.mean(lbvFTD2,0)+np.std(lbvFTD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    l_pd = axlist[1][2].fill_between(steps, np.mean(lPD2,0)-np.std(lPD2,0)/math.sqrt(cntExperiments),np.mean(lPD2,0)+np.std(lPD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
    l_m = axlist[1][2].fill_between(steps, np.mean(lM2,0)-np.std(lM2,0)/math.sqrt(cntExperiments),np.mean(lM2,0)+np.std(lM2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')

    l_add = axlist[1][3].fill_between(steps, np.mean(rADD2,0)-np.std(rADD2,0)/math.sqrt(cntExperiments),np.mean(rADD2,0)+np.std(rADD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='black')
    l_adhd = axlist[1][3].fill_between(steps, np.mean(rADHD2,0)-np.std(rADHD2,0)/math.sqrt(cntExperiments),np.mean(rADHD2,0)+np.std(rADHD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='blue')
    l_ad = axlist[1][3].fill_between(steps, np.mean(rAD2,0)-np.std(rAD2,0)/math.sqrt(cntExperiments),np.mean(rAD2,0)+np.std(rAD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='orange')
    l_cp = axlist[1][3].fill_between(steps, np.mean(rCP2,0)-np.std(rCP2,0)/math.sqrt(cntExperiments),np.mean(rCP2,0)+np.std(rCP2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='red')
    l_bvftd = axlist[1][3].fill_between(steps, np.mean(rbvFTD2,0)-np.std(rbvFTD2,0)/math.sqrt(cntExperiments),np.mean(rbvFTD2,0)+np.std(rbvFTD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='green')
    l_pd = axlist[1][3].fill_between(steps, np.mean(rPD2,0)-np.std(rPD2,0)/math.sqrt(cntExperiments),np.mean(rPD2,0)+np.std(rPD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='purple')
    l_m = axlist[1][3].fill_between(steps, np.mean(rM2,0)-np.std(rM2,0)/math.sqrt(cntExperiments),np.mean(rM2,0)+np.std(rM2,0)/math.sqrt(cntExperiments),alpha=alpha_plot, color='brown')

    fig.tight_layout() 
    fig.savefig(fig_name)
    
    return fig, np.mean(cADD,0)[-1],np.mean(cADHD,0)[-1],np.mean(cAD,0)[-1],np.mean(cCP,0)[-1],np.mean(cbvFTD,0)[-1],np.mean(cPD,0)[-1],np.mean(cM,0)[-1],reportADD["reward"],reportADHD["reward"],reportAD["reward"],reportCP["reward"],reportbvFTD["reward"],reportPD["reward"],reportM["reward"],reportADD["actions"],reportADHD["actions"],reportAD["actions"],reportCP["actions"],reportbvFTD["actions"],reportPD["actions"],reportM["actions"]

# draw graphs of different algorithms in IGT
def drawGraphIGT_mental(reportADD, reportADHD, reportAD,reportCP,reportbvFTD,reportPD,reportM,reward_from_A,reward_from_B,reward_from_C,reward_from_D,fig_name,draw_Exponential=False,plotShortTerm=False,shortTerm=100):
    plt.rcParams['figure.figsize'] = [20, 10]
    lw = 2
    
    yADD,yADHD,yAD,yCP,ybvFTD,yPD,yM = reportADD['percent'],reportADHD['percent'],reportAD['percent'],reportCP['percent'],reportbvFTD['percent'],reportPD['percent'],reportM['percent']
    yADD,yADHD,yAD,yCP,ybvFTD,yPD,yM = 100-yADD,100-yADHD,100-yAD,100-yCP,100-ybvFTD,100-yPD,100-yM
    aADD1,aADHD1,aAD1,aCP1,abvFTD1,aPD1,aM1 = reportADD["Q1(I)a"],reportADHD["Q1(I)a"],reportAD["Q1(I)a"],reportCP["Q1(I)a"],reportbvFTD["Q1(I)a"],reportPD["Q1(I)a"],reportM["Q1(I)a"]
    bADD1,bADHD1,bAD1,bCP1,bbvFTD1,bPD1,bM1 = reportADD["Q1(I)b"],reportADHD["Q1(I)b"],reportAD["Q1(I)b"],reportCP["Q1(I)b"],reportbvFTD["Q1(I)b"],reportPD["Q1(I)b"],reportM["Q1(I)b"]
    cADD1,cADHD1,cAD1,cCP1,cbvFTD1,cPD1,cM1 = reportADD["Q1(I)c"],reportADHD["Q1(I)c"],reportAD["Q1(I)c"],reportCP["Q1(I)c"],reportbvFTD["Q1(I)c"],reportPD["Q1(I)c"],reportM["Q1(I)c"]
    dADD1,dADHD1,dAD1,dCP1,dbvFTD1,dPD1,dM1 = reportADD["Q1(I)d"],reportADHD["Q1(I)d"],reportAD["Q1(I)d"],reportCP["Q1(I)d"],reportbvFTD["Q1(I)d"],reportPD["Q1(I)d"],reportM["Q1(I)d"]
    aADD2,aADHD2,aAD2,aCP2,abvFTD2,aPD2,aM2 = reportADD["Q2(I)a"],reportADHD["Q2(I)a"],reportAD["Q2(I)a"],reportCP["Q2(I)a"],reportbvFTD["Q2(I)a"],reportPD["Q2(I)a"],reportM["Q2(I)a"]
    bADD2,bADHD2,bAD2,bCP2,bbvFTD2,bPD2,bM2 = reportADD["Q2(I)b"],reportADHD["Q2(I)b"],reportAD["Q2(I)b"],reportCP["Q2(I)b"],reportbvFTD["Q2(I)b"],reportPD["Q2(I)b"],reportM["Q2(I)b"]
    cADD2,cADHD2,cAD2,cCP2,cbvFTD2,cPD2,cM2 = reportADD["Q2(I)c"],reportADHD["Q2(I)c"],reportAD["Q2(I)c"],reportCP["Q2(I)c"],reportbvFTD["Q2(I)c"],reportPD["Q2(I)c"],reportM["Q2(I)c"]
    dADD2,dADHD2,dAD2,dCP2,dbvFTD2,dPD2,dM2 = reportADD["Q2(I)d"],reportADHD["Q2(I)d"],reportAD["Q2(I)d"],reportCP["Q2(I)d"],reportbvFTD["Q2(I)d"],reportPD["Q2(I)d"],reportM["Q2(I)d"]
    rADD,rADHD,rAD,rCP,rbvFTD,rPD,rM = np.cumsum(reportADD["reward"],1),np.cumsum(reportADHD["reward"],1),np.cumsum(reportAD["reward"],1),np.cumsum(reportCP["reward"],1),np.cumsum(reportbvFTD["reward"],1),np.cumsum(reportPD["reward"],1),np.cumsum(reportM["reward"],1)
  
    steps = np.arange(yADD.shape[1])+1

    fig = plt.figure()    
    fig.subplots_adjust(hspace=0.5)

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

    data = []
    for i in range(1000):
        a = np.sum(reward_from_A(i))
        b = np.sum(reward_from_B(i))
        c = np.sum(reward_from_C(i))
        d = np.sum(reward_from_D(i))
        data.append([a,b,c,d])
    data = pd.DataFrame(data, columns=['A','B','C','D'])
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
        
        l_add, = axlist[0][1].plot(steps[:shortTerm], np.mean(yADD,0)[:shortTerm], marker='', color='black', linewidth=lw, label="ADD")
        l_adhd, = axlist[0][1].plot(steps[:shortTerm], np.mean(yADHD,0)[:shortTerm], marker='', color='blue',linewidth=lw, label="ADHD")
        l_ad, = axlist[0][1].plot(steps[:shortTerm], np.mean(yAD,0)[:shortTerm], marker='', color='orange',linewidth=lw, label="AD")
        l_cp, = axlist[0][1].plot(steps[:shortTerm], np.mean(yCP,0)[:shortTerm], marker='', color='red',linewidth=lw, label="CP")
        l_bvftd, = axlist[0][1].plot(steps[:shortTerm], np.mean(ybvFTD,0)[:shortTerm], marker='', color='green',linewidth=lw, label="bvFTD")
        l_pd, = axlist[0][1].plot(steps[:shortTerm], np.mean(yPD,0)[:shortTerm], marker='', color='purple',linewidth=lw, label="PD")
        l_m, = axlist[0][1].plot(steps[:shortTerm], np.mean(yM,0)[:shortTerm], marker='', color='brown',linewidth=lw, label="M")
        axlist[0][1].set_xlabel('Episodes')
        axlist[0][1].set_ylabel('% choosing better decks (short-term)')
        axlist[0][1].set_ylim(0,100)
        axlist[0][1].grid(True)

        last = 10
        l_add, = axlist[0][2].plot(steps[-last:], np.mean(yADD,0)[-last:], marker='', color='black', linewidth=lw, label="ADD")
        l_adhd, = axlist[0][2].plot(steps[-last:], np.mean(yADHD,0)[-last:], marker='', color='blue',linewidth=lw, label="ADHD")
        l_ad, = axlist[0][2].plot(steps[-last:], np.mean(yAD,0)[-last:], marker='', color='orange',linewidth=lw, label="AD")
        l_cp, = axlist[0][2].plot(steps[-last:], np.mean(yCP,0)[-last:], marker='', color='red',linewidth=lw, label="CP")
        l_bvftd, = axlist[0][2].plot(steps[-last:], np.mean(ybvFTD,0)[-last:], marker='', color='green',linewidth=lw, label="bvFTD")
        l_pd, = axlist[0][2].plot(steps[-last:], np.mean(yPD,0)[-last:], marker='', color='purple',linewidth=lw, label="PD")
        l_m, = axlist[0][2].plot(steps[-last:], np.mean(yM,0)[-last:], marker='', color='brown',linewidth=lw, label="M")
#         axlist[0][2].set_xlabel('Episodes')
        axlist[0][2].set_ylabel('% choosing better decks (long-term)')
#         axlist[0][2].get_yaxis().set_visible(False)
#         axlist[0][2].yaxis('off')
#         axlist[0][2].yaxis.tick_right()
        axlist[0][2].set_ylim(0,100)
        axlist[0][2].grid(True)
        
        l_add, = axlist[0][3].plot(steps[:shortTerm], np.mean(rADD,0)[:shortTerm], marker='', color='black', linewidth=lw, label="ADD")
        l_adhd, = axlist[0][3].plot(steps[:shortTerm], np.mean(rADHD,0)[:shortTerm], marker='', color='blue',linewidth=lw, label="ADHD")
        l_ad, = axlist[0][3].plot(steps[:shortTerm], np.mean(rAD,0)[:shortTerm], marker='', color='orange',linewidth=lw, label="AD")
        l_cp, = axlist[0][3].plot(steps[:shortTerm], np.mean(rCP,0)[:shortTerm], marker='', color='red',linewidth=lw, label="CP")
        l_bvftd, = axlist[0][3].plot(steps[:shortTerm], np.mean(rbvFTD,0)[:shortTerm], marker='', color='green',linewidth=lw, label="bvFTD")
        l_pd, = axlist[0][3].plot(steps[:shortTerm], np.mean(rPD,0)[:shortTerm], marker='', color='purple',linewidth=lw, label="PD")
        l_m, = axlist[0][3].plot(steps[:shortTerm], np.mean(rM,0)[:shortTerm], marker='', color='brown',linewidth=lw, label="M")
        axlist[0][3].set_xlabel('Episodes')
        axlist[0][3].set_ylabel('Cumulative episode rewards')
        axlist[0][3].legend()
        axlist[0][3].grid(True)
        
        l_add, = axlist[0][4].plot(steps[-last:], np.mean(rADD,0)[-last:], marker='', color='black', linewidth=lw, label="ADD")
        l_adhd, = axlist[0][4].plot(steps[-last:], np.mean(rADHD,0)[-last:], marker='', color='blue',linewidth=lw, label="ADHD")
        l_ad, = axlist[0][4].plot(steps[-last:], np.mean(rAD,0)[-last:], marker='', color='orange',linewidth=lw, label="AD")
        l_cp, = axlist[0][4].plot(steps[-last:], np.mean(rCP,0)[-last:], marker='', color='red',linewidth=lw, label="CP")
        l_bvftd, = axlist[0][4].plot(steps[-last:], np.mean(rbvFTD,0)[-last:], marker='', color='green',linewidth=lw, label="bvFTD")
        l_pd, = axlist[0][4].plot(steps[-last:], np.mean(rPD,0)[-last:], marker='', color='purple',linewidth=lw, label="PD")
        l_m, = axlist[0][4].plot(steps[-last:], np.mean(rM,0)[-last:], marker='', color='brown',linewidth=lw, label="M")
#         axlist[0][4].set_xlabel('Episodes')
#         axlist[0][4].set_ylabel('Cumulative episode rewards')
#         axlist[0][4].yaxis.tick_right()
        axlist[0][4].grid(True)
        
    else:
        
        l_add, = axlist[0][1].plot(steps, np.mean(yADD,0) , marker='', color='black', linewidth=lw, label="ADD")
        l_adhd, = axlist[0][1].plot(steps, np.mean(yADHD,0), marker='', color='blue',linewidth=lw, label="ADHD")
        l_ad, = axlist[0][1].plot(steps, np.mean(yAD,0), marker='', color='orange',linewidth=lw, label="AD")
        l_cp, = axlist[0][1].plot(steps, np.mean(yCP,0), marker='', color='red',linewidth=lw, label="CP")
        l_bvftd, = axlist[0][1].plot(steps, np.mean(ybvFTD,0), marker='', color='green',linewidth=lw, label="bvFTD")
        l_pd, = axlist[0][1].plot(steps, np.mean(yPD,0), marker='', color='purple',linewidth=lw, label="PD")
        l_m, = axlist[0][1].plot(steps, np.mean(yM,0), marker='', color='brown',linewidth=lw, label="M")
        axlist[0][1].set_xlabel('Episodes')
        axlist[0][1].set_ylabel('% choosing better decks')
        axlist[0][1].set_ylim(0,100)
        axlist[0][1].grid(True)

        l_add, = axlist[0][2].plot(steps, np.mean(rADD,0) , marker='', color='black', linewidth=lw, label="ADD")
        l_adhd, = axlist[0][2].plot(steps, np.mean(rADHD,0), marker='', color='blue',linewidth=lw, label="ADHD")
        l_ad, = axlist[0][2].plot(steps, np.mean(rAD,0), marker='', color='orange',linewidth=lw, label="AD")
        l_cp, = axlist[0][2].plot(steps, np.mean(rCP,0), marker='', color='red',linewidth=lw, label="CP")
        l_bvftd, = axlist[0][2].plot(steps, np.mean(rbvFTD,0), marker='', color='green',linewidth=lw, label="bvFTD")
        l_pd, = axlist[0][2].plot(steps, np.mean(rPD,0), marker='', color='purple',linewidth=lw, label="PD")
        l_m, = axlist[0][2].plot(steps, np.mean(rM,0), marker='', color='brown',linewidth=lw, label="M")
        axlist[0][2].set_xlabel('Episodes')
        axlist[0][2].set_ylabel('Cumulative episode rewards')
        axlist[0][2].grid(True)
    
    l_add, = axlist[1][0].plot(steps, np.mean(aADD1,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[1][0].plot(steps, np.mean(aADHD1,0), marker='', color='blue',linewidth=lw, label="ADHD")
    l_ad, = axlist[1][0].plot(steps, np.mean(aAD1,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[1][0].plot(steps, np.mean(aCP1,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[1][0].plot(steps, np.mean(abvFTD1,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[1][0].plot(steps, np.mean(aPD1,0), marker='', color='purple',linewidth=lw, label="PD")
    l_m, = axlist[1][0].plot(steps, np.mean(aM1,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[1][0].set_xlabel('Episodes')
    axlist[1][0].set_ylabel('Q1 for picking A')
    axlist[1][0].grid(True)
 
    l_add, = axlist[1][1].plot(steps, np.mean(bADD1,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[1][1].plot(steps, np.mean(bADHD1,0), marker='', color='blue',linewidth=lw, label="ADHD")
    l_ad, = axlist[1][1].plot(steps, np.mean(bAD1,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[1][1].plot(steps, np.mean(bCP1,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[1][1].plot(steps, np.mean(bbvFTD1,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[1][1].plot(steps, np.mean(bPD1,0), marker='', color='purple',linewidth=lw, label="PD")
    l_m, = axlist[1][1].plot(steps, np.mean(bM1,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[1][1].set_xlabel('Episodes')
    axlist[1][1].set_ylabel('Q1 for picking B')
    axlist[1][1].grid(True)

    l_add, = axlist[1][2].plot(steps, np.mean(cADD1,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[1][2].plot(steps, np.mean(cADHD1,0), marker='', color='blue',linewidth=lw, label="ADHD")
    l_ad, = axlist[1][2].plot(steps, np.mean(cAD1,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[1][2].plot(steps, np.mean(cCP1,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[1][2].plot(steps, np.mean(cbvFTD1,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[1][2].plot(steps, np.mean(cPD1,0), marker='', color='purple',linewidth=lw, label="PD")
    l_m, = axlist[1][2].plot(steps, np.mean(cM1,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[1][2].set_xlabel('Episodes')
    axlist[1][2].set_ylabel('Q1 for picking C')
    axlist[1][2].grid(True)
 
    l_add, = axlist[1][3].plot(steps, np.mean(dADD1,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[1][3].plot(steps, np.mean(dADHD1,0), marker='', color='blue',linewidth=lw, label="ADHD")
    l_ad, = axlist[1][3].plot(steps, np.mean(dAD1,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[1][3].plot(steps, np.mean(dCP1,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[1][3].plot(steps, np.mean(dbvFTD1,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[1][3].plot(steps, np.mean(dPD1,0), marker='', color='purple',linewidth=lw, label="PD")
    l_m, = axlist[1][3].plot(steps, np.mean(dM1,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[1][3].set_xlabel('Episodes')
    axlist[1][3].set_ylabel('Q1 for picking D')
    axlist[1][3].grid(True)
 
     
    l_add, = axlist[2][0].plot(steps, np.mean(aADD2,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[2][0].plot(steps, np.mean(aADHD2,0), marker='', color='blue', linewidth=lw, label="ADHD")
    l_ad, = axlist[2][0].plot(steps, np.mean(aAD2,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[2][0].plot(steps, np.mean(aCP2,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[2][0].plot(steps, np.mean(abvFTD2,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[2][0].plot(steps, np.mean(aPD2,0), marker='', color='purple', linewidth=lw, label="PD")
    l_m, = axlist[2][0].plot(steps, np.mean(aM2,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[2][0].set_xlabel('Episodes')
    axlist[2][0].set_ylabel('Q2 for picking A')
    axlist[2][0].grid(True)
 
    l_add, = axlist[2][1].plot(steps, np.mean(bADD2,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[2][1].plot(steps, np.mean(bADHD2,0), marker='', color='blue', linewidth=lw, label="ADHD")
    l_ad, = axlist[2][1].plot(steps, np.mean(bAD2,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[2][1].plot(steps, np.mean(bCP2,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[2][1].plot(steps, np.mean(bbvFTD2,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[2][1].plot(steps, np.mean(bPD2,0), marker='', color='purple', linewidth=lw, label="PD")
    l_m, = axlist[2][1].plot(steps, np.mean(bM2,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[2][1].set_xlabel('Episodes')
    axlist[2][1].set_ylabel('Q2 for picking B')
    axlist[2][1].grid(True)

    l_add, = axlist[2][2].plot(steps, np.mean(cADD2,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[2][2].plot(steps, np.mean(cADHD2,0), marker='', color='blue', linewidth=lw, label="ADHD")
    l_ad, = axlist[2][2].plot(steps, np.mean(cAD2,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[2][2].plot(steps, np.mean(cCP2,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[2][2].plot(steps, np.mean(cbvFTD2,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[2][2].plot(steps, np.mean(cPD2,0), marker='', color='purple', linewidth=lw, label="PD")
    l_m, = axlist[2][2].plot(steps, np.mean(cM2,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[2][2].set_xlabel('Episodes')
    axlist[2][2].set_ylabel('Q2 for picking C')
    axlist[2][2].grid(True)
 
    l_add, = axlist[2][3].plot(steps, np.mean(dADD2,0) , marker='', color='black', linewidth=lw, label="ADD")
    l_adhd, = axlist[2][3].plot(steps, np.mean(dADHD2,0), marker='', color='blue', linewidth=lw, label="ADHD")
    l_ad, = axlist[2][3].plot(steps, np.mean(dAD2,0), marker='', color='orange',linewidth=lw, label="AD")
    l_cp, = axlist[2][3].plot(steps, np.mean(dCP2,0), marker='', color='red',linewidth=lw, label="CP")
    l_bvftd, = axlist[2][3].plot(steps, np.mean(dbvFTD2,0), marker='', color='green',linewidth=lw, label="bvFTD")
    l_pd, = axlist[2][3].plot(steps, np.mean(dPD2,0), marker='', color='purple', linewidth=lw, label="PD")
    l_m, = axlist[2][3].plot(steps, np.mean(dM2,0), marker='', color='brown',linewidth=lw, label="M")
    axlist[2][3].set_xlabel('Episodes')
    axlist[2][3].set_ylabel('Q2 for picking D')
    axlist[2][3].legend()
    axlist[2][3].grid(True)   

    alpha_plot = 0.2
    
    if plotShortTerm:
        
        last = 10
        l_add= axlist[0][1].fill_between(steps[:shortTerm], np.mean(yADD,0)[:shortTerm]-np.std(yADD,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(yADD,0)[:shortTerm]+np.std(yADD,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
        l_adhd= axlist[0][1].fill_between(steps[:shortTerm], np.mean(yADHD,0)[:shortTerm]-np.std(yADHD,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(yADHD,0)[:shortTerm]+np.std(yADHD,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
        l_ad= axlist[0][1].fill_between(steps[:shortTerm], np.mean(yAD,0)[:shortTerm]-np.std(yAD,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(yAD,0)[:shortTerm]+np.std(yAD,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
        l_cp= axlist[0][1].fill_between(steps[:shortTerm], np.mean(yCP,0)[:shortTerm]-np.std(yCP,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(yCP,0)[:shortTerm]+np.std(yCP,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
        l_bvftd= axlist[0][1].fill_between(steps[:shortTerm], np.mean(ybvFTD,0)[:shortTerm]-np.std(ybvFTD,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(ybvFTD,0)[:shortTerm]+np.std(ybvFTD,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
        l_pd= axlist[0][1].fill_between(steps[:shortTerm], np.mean(yPD,0)[:shortTerm]-np.std(yPD,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(yPD,0)[:shortTerm]+np.std(yPD,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_m= axlist[0][1].fill_between(steps[:shortTerm], np.mean(yM,0)[:shortTerm]-np.std(yM,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(yM,0)[:shortTerm]+np.std(yM,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

        l_add= axlist[0][2].fill_between(steps[-last:], np.mean(yADD,0)[-last:]-np.std(yADD,0)[-last:]/math.sqrt(cntExperiments),np.mean(yADD,0)[-last:]+np.std(yADD,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
        l_adhd= axlist[0][2].fill_between(steps[-last:], np.mean(yADHD,0)[-last:]-np.std(yADHD,0)[-last:]/math.sqrt(cntExperiments),np.mean(yADHD,0)[-last:]+np.std(yADHD,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
        l_ad= axlist[0][2].fill_between(steps[-last:], np.mean(yAD,0)[-last:]-np.std(yAD,0)[-last:]/math.sqrt(cntExperiments),np.mean(yAD,0)[-last:]+np.std(yAD,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
        l_cp= axlist[0][2].fill_between(steps[-last:], np.mean(yCP,0)[-last:]-np.std(yCP,0)[-last:]/math.sqrt(cntExperiments),np.mean(yCP,0)[-last:]+np.std(yCP,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
        l_bvftd= axlist[0][2].fill_between(steps[-last:], np.mean(ybvFTD,0)[-last:]-np.std(ybvFTD,0)[-last:]/math.sqrt(cntExperiments),np.mean(ybvFTD,0)[-last:]+np.std(ybvFTD,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
        l_pd= axlist[0][2].fill_between(steps[-last:], np.mean(yPD,0)[-last:]-np.std(yPD,0)[-last:]/math.sqrt(cntExperiments),np.mean(yPD,0)[-last:]+np.std(yPD,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_m= axlist[0][2].fill_between(steps[-last:], np.mean(yM,0)[-last:]-np.std(yM,0)[-last:]/math.sqrt(cntExperiments),np.mean(yM,0)[-last:]+np.std(yM,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

        l_add= axlist[0][3].fill_between(steps[:shortTerm], np.mean(rADD,0)[:shortTerm]-np.std(rADD,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(rADD,0)[:shortTerm]+np.std(rADD,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
        l_adhd= axlist[0][3].fill_between(steps[:shortTerm], np.mean(rADHD,0)[:shortTerm]-np.std(rADHD,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(rADHD,0)[:shortTerm]+np.std(rADHD,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
        l_ad= axlist[0][3].fill_between(steps[:shortTerm], np.mean(rAD,0)[:shortTerm]-np.std(rAD,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(rAD,0)[:shortTerm]+np.std(rAD,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
        l_cp= axlist[0][3].fill_between(steps[:shortTerm], np.mean(rCP,0)[:shortTerm]-np.std(rCP,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(rCP,0)[:shortTerm]+np.std(rCP,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
        l_bvftd= axlist[0][3].fill_between(steps[:shortTerm], np.mean(rbvFTD,0)[:shortTerm]-np.std(rbvFTD,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(rbvFTD,0)[:shortTerm]+np.std(rbvFTD,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
        l_pd= axlist[0][3].fill_between(steps[:shortTerm], np.mean(rPD,0)[:shortTerm]-np.std(rPD,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(rPD,0)[:shortTerm]+np.std(rPD,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_m= axlist[0][3].fill_between(steps[:shortTerm], np.mean(rM,0)[:shortTerm]-np.std(rM,0)[:shortTerm]/math.sqrt(cntExperiments),np.mean(rM,0)[:shortTerm]+np.std(rM,0)[:shortTerm]/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

        l_add= axlist[0][4].fill_between(steps[-last:], np.mean(rADD,0)[-last:]-np.std(rADD,0)[-last:]/math.sqrt(cntExperiments),np.mean(rADD,0)[-last:]+np.std(rADD,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
        l_adhd= axlist[0][4].fill_between(steps[-last:], np.mean(rADHD,0)[-last:]-np.std(rADHD,0)[-last:]/math.sqrt(cntExperiments),np.mean(rADHD,0)[-last:]+np.std(rADHD,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
        l_ad= axlist[0][4].fill_between(steps[-last:], np.mean(rAD,0)[-last:]-np.std(rAD,0)[-last:]/math.sqrt(cntExperiments),np.mean(rAD,0)[-last:]+np.std(rAD,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
        l_cp= axlist[0][4].fill_between(steps[-last:], np.mean(rCP,0)[-last:]-np.std(rCP,0)[-last:]/math.sqrt(cntExperiments),np.mean(rCP,0)[-last:]+np.std(rCP,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
        l_bvftd= axlist[0][4].fill_between(steps[-last:], np.mean(rbvFTD,0)[-last:]-np.std(rbvFTD,0)[-last:]/math.sqrt(cntExperiments),np.mean(rbvFTD,0)[-last:]+np.std(rbvFTD,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
        l_pd= axlist[0][4].fill_between(steps[-last:], np.mean(rPD,0)[-last:]-np.std(rPD,0)[-last:]/math.sqrt(cntExperiments),np.mean(rPD,0)[-last:]+np.std(rPD,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_m= axlist[0][4].fill_between(steps[-last:], np.mean(rM,0)[-last:]-np.std(rM,0)[-last:]/math.sqrt(cntExperiments),np.mean(rM,0)[-last:]+np.std(rM,0)[-last:]/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
    
    else:
        
        l_add= axlist[0][1].fill_between(steps, np.mean(yADD,0)-np.std(yADD,0)/math.sqrt(cntExperiments),np.mean(yADD,0)+np.std(yADD,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
        l_adhd= axlist[0][1].fill_between(steps, np.mean(yADHD,0)-np.std(yADHD,0)/math.sqrt(cntExperiments),np.mean(yADHD,0)+np.std(yADHD,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
        l_ad= axlist[0][1].fill_between(steps, np.mean(yAD,0)-np.std(yAD,0)/math.sqrt(cntExperiments),np.mean(yAD,0)+np.std(yAD,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
        l_cp= axlist[0][1].fill_between(steps, np.mean(yCP,0)-np.std(yCP,0)/math.sqrt(cntExperiments),np.mean(yCP,0)+np.std(yCP,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
        l_bvftd= axlist[0][1].fill_between(steps, np.mean(ybvFTD,0)-np.std(ybvFTD,0)/math.sqrt(cntExperiments),np.mean(ybvFTD,0)+np.std(ybvFTD,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
        l_pd= axlist[0][1].fill_between(steps, np.mean(yPD,0)-np.std(yPD,0)/math.sqrt(cntExperiments),np.mean(yPD,0)+np.std(yPD,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_m= axlist[0][1].fill_between(steps, np.mean(yM,0)-np.std(yM,0)/math.sqrt(cntExperiments),np.mean(yM,0)+np.std(yM,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

        l_add= axlist[0][2].fill_between(steps, np.mean(rADD,0)-np.std(rADD,0)/math.sqrt(cntExperiments),np.mean(rADD,0)+np.std(rADD,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
        l_adhd= axlist[0][2].fill_between(steps, np.mean(rADHD,0)-np.std(rADHD,0)/math.sqrt(cntExperiments),np.mean(rADHD,0)+np.std(rADHD,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
        l_ad= axlist[0][2].fill_between(steps, np.mean(rAD,0)-np.std(rAD,0)/math.sqrt(cntExperiments),np.mean(rAD,0)+np.std(rAD,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
        l_cp= axlist[0][2].fill_between(steps, np.mean(rCP,0)-np.std(rCP,0)/math.sqrt(cntExperiments),np.mean(rCP,0)+np.std(rCP,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
        l_bvftd= axlist[0][2].fill_between(steps, np.mean(rbvFTD,0)-np.std(rbvFTD,0)/math.sqrt(cntExperiments),np.mean(rbvFTD,0)+np.std(rbvFTD,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
        l_pd= axlist[0][2].fill_between(steps, np.mean(rPD,0)-np.std(rPD,0)/math.sqrt(cntExperiments),np.mean(rPD,0)+np.std(rPD,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
        l_m= axlist[0][2].fill_between(steps, np.mean(rM,0)-np.std(rM,0)/math.sqrt(cntExperiments),np.mean(rM,0)+np.std(rM,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
 
          
    l_add= axlist[1][0].fill_between(steps, np.mean(aADD1,0)-np.std(aADD1,0)/math.sqrt(cntExperiments),np.mean(aADD1,0)+np.std(aADD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_adhd= axlist[1][0].fill_between(steps, np.mean(aADHD1,0)-np.std(aADHD1,0)/math.sqrt(cntExperiments),np.mean(aADHD1,0)+np.std(aADHD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_ad= axlist[1][0].fill_between(steps, np.mean(aAD1,0)-np.std(aAD1,0)/math.sqrt(cntExperiments),np.mean(aAD1,0)+np.std(aAD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_cp= axlist[1][0].fill_between(steps, np.mean(aCP1,0)-np.std(aCP1,0)/math.sqrt(cntExperiments),np.mean(aCP1,0)+np.std(aCP1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_bvftd= axlist[1][0].fill_between(steps, np.mean(abvFTD1,0)-np.std(abvFTD1,0)/math.sqrt(cntExperiments),np.mean(abvFTD1,0)+np.std(abvFTD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    l_pd= axlist[1][0].fill_between(steps, np.mean(aPD1,0)-np.std(aPD1,0)/math.sqrt(cntExperiments),np.mean(aPD1,0)+np.std(aPD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    l_m= axlist[1][0].fill_between(steps, np.mean(aM1,0)-np.std(aM1,0)/math.sqrt(cntExperiments),np.mean(aM1,0)+np.std(aM1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
 
    l_add= axlist[1][1].fill_between(steps, np.mean(bADD1,0)-np.std(bADD1,0)/math.sqrt(cntExperiments),np.mean(bADD1,0)+np.std(bADD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_adhd= axlist[1][1].fill_between(steps, np.mean(bADHD1,0)-np.std(bADHD1,0)/math.sqrt(cntExperiments),np.mean(bADHD1,0)+np.std(bADHD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_ad= axlist[1][1].fill_between(steps, np.mean(bAD1,0)-np.std(bAD1,0)/math.sqrt(cntExperiments),np.mean(bAD1,0)+np.std(bAD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_cp= axlist[1][1].fill_between(steps, np.mean(bCP1,0)-np.std(bCP1,0)/math.sqrt(cntExperiments),np.mean(bCP1,0)+np.std(bCP1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_bvftd= axlist[1][1].fill_between(steps, np.mean(bbvFTD1,0)-np.std(bbvFTD1,0)/math.sqrt(cntExperiments),np.mean(bbvFTD1,0)+np.std(bbvFTD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    l_pd= axlist[1][1].fill_between(steps, np.mean(bPD1,0)-np.std(bPD1,0)/math.sqrt(cntExperiments),np.mean(bPD1,0)+np.std(bPD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    l_m= axlist[1][1].fill_between(steps, np.mean(bM1,0)-np.std(bM1,0)/math.sqrt(cntExperiments),np.mean(bM1,0)+np.std(bM1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

    l_add= axlist[1][2].fill_between(steps, np.mean(cADD1,0)-np.std(cADD1,0)/math.sqrt(cntExperiments),np.mean(cADD1,0)+np.std(cADD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_adhd= axlist[1][2].fill_between(steps, np.mean(cADHD1,0)-np.std(cADHD1,0)/math.sqrt(cntExperiments),np.mean(cADHD1,0)+np.std(cADHD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_ad= axlist[1][2].fill_between(steps, np.mean(cAD1,0)-np.std(cAD1,0)/math.sqrt(cntExperiments),np.mean(cAD1,0)+np.std(cAD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_cp= axlist[1][2].fill_between(steps, np.mean(cCP1,0)-np.std(cCP1,0)/math.sqrt(cntExperiments),np.mean(cCP1,0)+np.std(cCP1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_bvftd= axlist[1][2].fill_between(steps, np.mean(cbvFTD1,0)-np.std(cbvFTD1,0)/math.sqrt(cntExperiments),np.mean(cbvFTD1,0)+np.std(cbvFTD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    l_pd= axlist[1][2].fill_between(steps, np.mean(cPD1,0)-np.std(cPD1,0)/math.sqrt(cntExperiments),np.mean(cPD1,0)+np.std(cPD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    l_m= axlist[1][2].fill_between(steps, np.mean(cM1,0)-np.std(cM1,0)/math.sqrt(cntExperiments),np.mean(cM1,0)+np.std(cM1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
 
    l_add= axlist[1][3].fill_between(steps, np.mean(dADD1,0)-np.std(dADD1,0)/math.sqrt(cntExperiments),np.mean(dADD1,0)+np.std(dADD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_adhd= axlist[1][3].fill_between(steps, np.mean(dADHD1,0)-np.std(dADHD1,0)/math.sqrt(cntExperiments),np.mean(dADHD1,0)+np.std(dADHD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_ad= axlist[1][3].fill_between(steps, np.mean(dAD1,0)-np.std(dAD1,0)/math.sqrt(cntExperiments),np.mean(dAD1,0)+np.std(dAD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_cp= axlist[1][3].fill_between(steps, np.mean(dCP1,0)-np.std(dCP1,0)/math.sqrt(cntExperiments),np.mean(dCP1,0)+np.std(dCP1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_bvftd= axlist[1][3].fill_between(steps, np.mean(dbvFTD1,0)-np.std(dbvFTD1,0)/math.sqrt(cntExperiments),np.mean(dbvFTD1,0)+np.std(dbvFTD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    l_pd= axlist[1][3].fill_between(steps, np.mean(dPD1,0)-np.std(dPD1,0)/math.sqrt(cntExperiments),np.mean(dPD1,0)+np.std(dPD1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    l_m= axlist[1][3].fill_between(steps, np.mean(dM1,0)-np.std(dM1,0)/math.sqrt(cntExperiments),np.mean(dM1,0)+np.std(dM1,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
     
    l_add= axlist[2][0].fill_between(steps, np.mean(aADD2,0)-np.std(aADD2,0)/math.sqrt(cntExperiments),np.mean(aADD2,0)+np.std(aADD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_adhd= axlist[2][0].fill_between(steps, np.mean(aADHD2,0)-np.std(aADHD2,0)/math.sqrt(cntExperiments),np.mean(aADHD2,0)+np.std(aADHD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_ad= axlist[2][0].fill_between(steps, np.mean(aAD2,0)-np.std(aAD2,0)/math.sqrt(cntExperiments),np.mean(aAD2,0)+np.std(aAD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_cp= axlist[2][0].fill_between(steps, np.mean(aCP2,0)-np.std(aCP2,0)/math.sqrt(cntExperiments),np.mean(aCP2,0)+np.std(aCP2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_bvftd= axlist[2][0].fill_between(steps, np.mean(abvFTD2,0)-np.std(abvFTD2,0)/math.sqrt(cntExperiments),np.mean(abvFTD2,0)+np.std(abvFTD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    l_pd= axlist[2][0].fill_between(steps, np.mean(aPD2,0)-np.std(aPD2,0)/math.sqrt(cntExperiments),np.mean(aPD2,0)+np.std(aPD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    l_m= axlist[2][0].fill_between(steps, np.mean(aM2,0)-np.std(aM2,0)/math.sqrt(cntExperiments),np.mean(aM2,0)+np.std(aM2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
 
    l_add= axlist[2][1].fill_between(steps, np.mean(bADD2,0)-np.std(bADD2,0)/math.sqrt(cntExperiments),np.mean(bADD2,0)+np.std(bADD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_adhd= axlist[2][1].fill_between(steps, np.mean(bADHD2,0)-np.std(bADHD2,0)/math.sqrt(cntExperiments),np.mean(bADHD2,0)+np.std(bADHD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_ad= axlist[2][1].fill_between(steps, np.mean(bAD2,0)-np.std(bAD2,0)/math.sqrt(cntExperiments),np.mean(bAD2,0)+np.std(bAD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_cp= axlist[2][1].fill_between(steps, np.mean(bCP2,0)-np.std(bCP2,0)/math.sqrt(cntExperiments),np.mean(bCP2,0)+np.std(bCP2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_bvftd= axlist[2][1].fill_between(steps, np.mean(bbvFTD2,0)-np.std(bbvFTD2,0)/math.sqrt(cntExperiments),np.mean(bbvFTD2,0)+np.std(bbvFTD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    l_pd= axlist[2][1].fill_between(steps, np.mean(bPD2,0)-np.std(bPD2,0)/math.sqrt(cntExperiments),np.mean(bPD2,0)+np.std(bPD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    l_m= axlist[2][1].fill_between(steps, np.mean(bM2,0)-np.std(bM2,0)/math.sqrt(cntExperiments),np.mean(bM2,0)+np.std(bM2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

    l_add= axlist[2][2].fill_between(steps, np.mean(cADD2,0)-np.std(cADD2,0)/math.sqrt(cntExperiments),np.mean(cADD2,0)+np.std(cADD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_adhd= axlist[2][2].fill_between(steps, np.mean(cADHD2,0)-np.std(cADHD2,0)/math.sqrt(cntExperiments),np.mean(cADHD2,0)+np.std(cADHD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_ad= axlist[2][2].fill_between(steps, np.mean(cAD2,0)-np.std(cAD2,0)/math.sqrt(cntExperiments),np.mean(cAD2,0)+np.std(cAD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_cp= axlist[2][2].fill_between(steps, np.mean(cCP2,0)-np.std(cCP2,0)/math.sqrt(cntExperiments),np.mean(cCP2,0)+np.std(cCP2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_bvftd= axlist[2][2].fill_between(steps, np.mean(cbvFTD2,0)-np.std(cbvFTD2,0)/math.sqrt(cntExperiments),np.mean(cbvFTD2,0)-np.std(cbvFTD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    l_pd= axlist[2][2].fill_between(steps, np.mean(cPD2,0)-np.std(cPD2,0)/math.sqrt(cntExperiments),np.mean(cPD2,0)+np.std(cPD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    l_m= axlist[2][2].fill_between(steps, np.mean(cM2,0)-np.std(cM2,0)/math.sqrt(cntExperiments),np.mean(cM2,0)+np.std(cM2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')
 
    l_add= axlist[2][3].fill_between(steps, np.mean(dADD2,0)-np.std(dADD2,0)/math.sqrt(cntExperiments),np.mean(dADD2,0)+np.std(dADD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='black')
    l_adhd= axlist[2][3].fill_between(steps, np.mean(dADHD2,0)-np.std(dADHD2,0)/math.sqrt(cntExperiments),np.mean(dADHD2,0)+np.std(dADHD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='blue')
    l_ad= axlist[2][3].fill_between(steps, np.mean(dAD2,0)-np.std(dAD2,0)/math.sqrt(cntExperiments),np.mean(dAD2,0)+np.std(dAD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='orange')
    l_cp= axlist[2][3].fill_between(steps, np.mean(dCP2,0)-np.std(dCP2,0)/math.sqrt(cntExperiments),np.mean(dCP2,0)+np.std(dCP2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='red')
    l_bvftd= axlist[2][3].fill_between(steps, np.mean(dbvFTD2,0)-np.std(dbvFTD2,0)/math.sqrt(cntExperiments),np.mean(dbvFTD2,0)+np.std(dbvFTD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='green')
    l_pd= axlist[2][3].fill_between(steps, np.mean(dPD2,0)-np.std(dPD2,0)/math.sqrt(cntExperiments),np.mean(dPD2,0)+np.std(dPD2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='purple')
    l_m= axlist[2][3].fill_between(steps, np.mean(dM2,0)-np.std(dM2,0)/math.sqrt(cntExperiments),np.mean(dM2,0)+np.std(dM2,0)/math.sqrt(cntExperiments),alpha=alpha_plot,color='brown')

    fig.tight_layout(pad=0., w_pad=0., h_pad=0.5) 
    fig.savefig(fig_name)
    
    return fig, np.mean(rADD,0)[-1],np.mean(rADHD,0)[-1],np.mean(rAD,0)[-1],np.mean(rCP,0)[-1],np.mean(rbvFTD,0)[-1],np.mean(rPD,0)[-1],np.mean(rM,0)[-1],reportADD["reward"],reportADHD["reward"],reportAD["reward"],reportCP["reward"],reportbvFTD["reward"],reportPD["reward"],reportM["reward"],reportADD["actions"],reportADHD["actions"],reportAD["actions"],reportCP["actions"],reportbvFTD["actions"],reportPD["actions"],reportM["actions"]


