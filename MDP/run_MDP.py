from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('should have a scenario arguments. e.g. 1-20, or 21-120 for randomly generated biomodal distribution, 121-220 for trimodal, 221-320 for 4-modal etc.')

    try:
        scenario_id = int(sys.argv[1])
    except:
        sys.exit('Wrong second argument')

    prefix = ''
    label_B,label_C,fig_name,fig_name_flipped,reward_from_B,reward_from_C,kld,crossentropy,sB,sC,sdiff = load_scenario(scenario_id,prefix=prefix)

    algorithms = ('QL','DQL','SARSA','SQL','SQL2','MP','PQL','NQL')
    names = ('QL','DQL','SARSA','SQL-alg1','SQL-alg2','MP','PQL','NQL')
    # init report variables that will hold all the results

    algorithms_m = ('AD','ADD','ADHD','bvFTD','CP','PD','M')
    names_m = ('AD','ADD','ADHD','bvFTD','CP','PD','M')
        
    reports, reports_m = [],[]
    
    nTrials=100
    T=500
    nAct=20
    
    reward_functions = (reward_from_B,reward_from_C)
    labels = (label_B,label_C)
    for alg in algorithms:
        case = MDP(alg,reward_functions,nTrials,T,nAct_B=nAct,nAct_C=nAct)
        rep = case.run()
        reports.append(rep)
    fig, scores, rewards, actions = plotAgents(reports,names,nTrials,reward_functions,labels,fig_name+'_1.png')

    for alg in algorithms_m:
        case = MDP(alg,reward_functions,nTrials,T,nAct_B=nAct,nAct_C=nAct)
        rep = case.run()
        reports_m.append(rep)
    fig, scores_m, rewards_m, actions_m = plotAgents(reports_m,names_m,nTrials,reward_functions,labels,fig_name+'_2.png')

    winner = getWinner(scores,names)

    f = open(prefix+"MDP_results.py", "a")

    f.write("\n")
    f.write("# config is nTrials:"+str(nTrials)+", T:"+str(T)+", nAct:"+str(nAct)+"\n")
    f.write("scenario.append("+str(scenario_id)+")\n")

#     f.write("kld.append("+str(kld)+")\n")
#     f.write("sB.append("+str(sB)+")\n")
#     f.write("sC.append("+str(sC)+")\n")
#     f.write("sdiff.append("+str(sdiff)+")\n")
#     f.write("crossentropy.append("+str(crossentropy)+")\n")
    f.write("winner.append('"+winner+"')\n")

    for i,alg in enumerate(algorithms):
        f.write(alg+".append("+str(scores[i])+")\n")
        # np.save("./output/r_"+alg+"_MDP_"+str(scenario_id)+".npy",rewards[i],allow_pickle=True)
        # np.save("./output/a_"+alg+"_MDP_"+str(scenario_id)+".npy",actions[i],allow_pickle=True)

    for i,alg in enumerate(algorithms_m):
        f.write(alg+".append("+str(scores_m[i])+")\n")
        # np.save("./output/r_"+alg+"_MDP_"+str(scenario_id)+".npy",rewards_m[i],allow_pickle=True)
        # np.save("./output/a_"+alg+"_MDP_"+str(scenario_id)+".npy",actions_m[i],allow_pickle=True)

    f.close()
    
  
