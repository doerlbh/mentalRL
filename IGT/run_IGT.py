from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('should have a scenario arguments. 1-2 for IGT scenario')
    try:
        scenario_id = int(sys.argv[1])
    except:
        sys.exit('Wrong second argument')

    prefix = ''
    fig_name,reward_from_A,reward_from_B,reward_from_C,reward_from_D = load_IGT(scenario_id,prefix=prefix)

    algorithms = ('QL','DQL','SARSA','SQL','SQL2','MP','PQL','NQL')
    names = ('QL','DQL','SARSA','SQL-alg1','SQL-alg2','MP','PQL','NQL')
    # init report variables that will hold all the results

    algorithms_m = ('AD','ADD','ADHD','bvFTD','CP','PD','M')
    names_m = ('AD','ADD','ADHD','bvFTD','CP','PD','M')
        
    reports, reports_m = [],[]
    
    nTrials=200
    T=500
    
    reward_functions = (reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    labels = ('A','B','C','D')
    
    for alg in algorithms:
        case = IGT(alg,reward_functions,nTrials,T)
        rep = case.run()
        reports.append(rep)
    fig, scores, rewards, actions = plotAgents(reports,names,nTrials,reward_functions,labels,fig_name+'_1.png',isIGT=True,plotShortTerm=True)

    for alg in algorithms_m:
        case = IGT(alg,reward_functions,nTrials,T)
        rep = case.run()
        reports_m.append(rep)
    fig, scores_m, rewards_m, actions_m = plotAgents(reports_m,names_m,nTrials,reward_functions,labels,fig_name+'_2.png',isIGT=True,plotShortTerm=True)

    winner = getWinner(scores,names)

    f = open(prefix+"IGT_results.py", "a")

    f.write("\n")
    f.write("# config is nTrials:"+str(nTrials)+", T:"+str(T)+"\n")
    f.write("scenario.append("+str(scenario_id)+")\n")

#     f.write("kld.append("+str(kld)+")\n")
#     f.write("sB.append("+str(sB)+")\n")
#     f.write("sC.append("+str(sC)+")\n")
#     f.write("sdiff.append("+str(sdiff)+")\n")
#     f.write("crossentropy.append("+str(crossentropy)+")\n")
    f.write("winner.append('"+winner+"')\n")

    for i,alg in enumerate(algorithms):
        f.write(alg+".append("+str(scores[i])+")\n")
        # np.save("./output/r_"+alg+"_IGT_"+str(scenario_id)+".npy",rewards[i],allow_pickle=True)
        # np.save("./output/a_"+alg+"_IGT_"+str(scenario_id)+".npy",actions[i],allow_pickle=True)

    for i,alg in enumerate(algorithms_m):
        f.write(alg+".append("+str(scores_m[i])+")\n")
        # np.save("./output/r_"+alg+"_IGT_"+str(scenario_id)+".npy",rewards_m[i],allow_pickle=True)
        # np.save("./output/a_"+alg+"_IGT_"+str(scenario_id)+".npy",actions_m[i],allow_pickle=True)

    f.close()
        
    with open(prefix+'IGT_wi_'+str(scenario_id)+'.csv', 'wb') as record:
        for rep in reports: np.savetxt(record, rep["pos_reward"], delimiter=",",fmt='%5.2f')
        for rep in reports_m: np.savetxt(record, rep["pos_reward"], delimiter=",",fmt='%5.2f')

    with open(prefix+'IGT_lo_'+str(scenario_id)+'.csv', 'wb') as record:
        for rep in reports: np.savetxt(record, rep["neg_reward"], delimiter=",",fmt='%5.2f')
        for rep in reports_m: np.savetxt(record, rep["neg_reward"], delimiter=",",fmt='%5.2f')
        
    with open(prefix+'IGT_gn_'+str(scenario_id)+'.csv', 'wb') as record:
        for rep in reports: np.savetxt(record, rep["reward"], delimiter=",",fmt='%5.2f')
        for rep in reports_m: np.savetxt(record, rep["reward"], delimiter=",",fmt='%5.2f')
        
    f = open(prefix+"IGT_index.csv", "a")
    for name in names: 
        for i in np.arange(nTrials): 
            f.write(name+"\n")
    for name in names_m: 
        for i in np.arange(nTrials): 
            f.write(name+"\n")
    f.close()
