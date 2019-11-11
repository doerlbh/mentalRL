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

    label_B,label_C,fig_name,fig_name_flipped,reward_from_B,reward_from_C,kld,crossentropy,sB,sC,sdiff = load_scenario(scenario_id)

   # Note: Split Q Learning is the Standard HQL.

    # init report variables that will hold all the results
    reportQL={}
    reportDoubleQL={}
    reportSplitQL={}
    reportPositiveQl={}
    reportNegativeQl={}
    reportExponentialDoubleQl={}
    reportExponentialSplitQl={}
    reportSARSA={}

    reportADD,reportADHD,reportAD,reportCP,reportbvFTD,reportPD,reportM={},{},{},{},{},{},{}

    runLearning('Q-Learning',reportQL,cntExperiments,reward_from_B,reward_from_C)
    runLearning('Double Q-Learning',reportDoubleQL,cntExperiments,reward_from_B,reward_from_C)
    runLearning('Split Q-Learning',reportSplitQL,cntExperiments,reward_from_B,reward_from_C)
    runLearning('Positive Q-Learning',reportPositiveQl,cntExperiments,reward_from_B,reward_from_C)
    runLearning('Negative Q-Learning',reportNegativeQl,cntExperiments,reward_from_B,reward_from_C)
    runLearning('Exponential Double Q-Learning',reportExponentialDoubleQl, cntExperiments,reward_from_B,reward_from_C)
    runLearning('Exponential Split Q-Learning',reportExponentialSplitQl, cntExperiments,reward_from_B,reward_from_C)
    runLearning('SARSA',reportSARSA, cntExperiments,reward_from_B,reward_from_C)

    runLearning('ADD',reportADD, cntExperiments,reward_from_B,reward_from_C)
    runLearning('ADHD',reportADHD, cntExperiments,reward_from_B,reward_from_C)
    runLearning('AD',reportAD, cntExperiments,reward_from_B,reward_from_C)
    runLearning('CP',reportCP, cntExperiments,reward_from_B,reward_from_C)
    runLearning('bvFTD',reportbvFTD, cntExperiments,reward_from_B,reward_from_C)
    runLearning('PD',reportPD, cntExperiments,reward_from_B,reward_from_C)
    runLearning('M',reportM, cntExperiments,reward_from_B,reward_from_C)

    # print graphs
    fig,ql,dql,sql,pql,nql,edql,esql,sarsa,r_ql,r_dql,r_sql,r_pql,r_nql,r_edql,r_esql,r_sarsa,a_ql,a_dql,a_sql,a_pql,a_nql,a_edql,a_esql,a_sarsa = drawGraph(reportQL,reportDoubleQL,reportSplitQL,reportPositiveQl,reportNegativeQl,reportExponentialDoubleQl,reportExponentialSplitQl,reportSARSA,reward_from_B,reward_from_C,label_B,label_C,fig_name+'_mental_1.png',draw_SARSA=True)
    fig,add,adhd,ad,cp,bvftd,pd,m,r_add,r_adhd,r_ad,r_cp,r_bvftd,r_pd,r_m,a_add,a_adhd,a_ad,a_cp,a_bvftd,a_pd,a_m = drawGraph_mental(reportADD,reportADHD,reportAD,reportCP,reportbvFTD,reportPD,reportM,reward_from_B,reward_from_C,label_B,label_C,fig_name+'_mental_2.png')

    scores = [ql,dql,sql,pql,nql,edql,esql,sarsa,add,adhd,ad,cp,bvftd,pd,m]
    names = ['QL','DQL','SQL','PQL','NQL','EDQL','ESQL','SARSA','ADD','ADHD','AD','CP','bvFTD','PD','M']
    winner = getWinner(scores,names)

    f = open("MDP_results.py", "a")

    f.write("\n")
    f.write("scenario.append("+str(scenario_id)+")\n")
    f.write("kld.append("+str(kld)+")\n")
    f.write("sB.append("+str(sB)+")\n")
    f.write("sC.append("+str(sC)+")\n")
    f.write("sdiff.append("+str(sdiff)+")\n")
    f.write("crossentropy.append("+str(crossentropy)+")\n")
    f.write("winner.append('"+winner+"')\n")

    f.write("QL.append("+str(ql)+")\n")
    f.write("DQL.append("+str(dql)+")\n")
    f.write("SQL.append("+str(sql)+")\n")
    f.write("PQL.append("+str(pql)+")\n")
    f.write("NQL.append("+str(nql)+")\n")
    f.write("EDQL.append("+str(edql)+")\n")
    f.write("ESQL.append("+str(esql)+")\n")
    f.write("SARSA.append("+str(sarsa)+")\n")
    
#     f.write("rQL.append("+str(r_ql)+")\n")
#     f.write("rDQL.append("+str(r_dql)+")\n")
#     f.write("rSQL.append("+str(r_sql)+")\n")
#     f.write("rPQL.append("+str(r_pql)+")\n")
#     f.write("rNQL.append("+str(r_nql)+")\n")
#     f.write("rEDQL.append("+str(r_edql)+")\n")
#     f.write("rESQL.append("+str(r_esql)+")\n")
#     f.write("rSARSA.append("+str(r_sarsa)+")\n")
#     
#     f.write("aQL.append("+str(a_ql)+")\n")
#     f.write("aDQL.append("+str(a_dql)+")\n")
#     f.write("aSQL.append("+str(a_sql)+")\n")
#     f.write("aPQL.append("+str(a_pql)+")\n")
#     f.write("aNQL.append("+str(a_nql)+")\n")
#     f.write("aEDQL.append("+str(a_edql)+")\n")
#     f.write("aESQL.append("+str(a_esql)+")\n")
#     f.write("aSARSA.append("+str(a_sarsa)+")\n")

    f.write("ADD.append("+str(add)+")\n")
    f.write("ADHD.append("+str(adhd)+")\n")
    f.write("AD.append("+str(ad)+")\n")
    f.write("CP.append("+str(cp)+")\n")
    f.write("bvFTD.append("+str(bvftd)+")\n")
    f.write("PD.append("+str(pd)+")\n")
    f.write("M.append("+str(m)+")\n")

#     f.write("rADD.append("+str(r_add)+")\n")
#     f.write("rADHD.append("+str(r_adhd)+")\n")
#     f.write("rAD.append("+str(r_ad)+")\n")
#     f.write("rCP.append("+str(r_cp)+")\n")
#     f.write("rbvFTD.append("+str(r_bvftd)+")\n")
#     f.write("rPD.append("+str(r_pd)+")\n")
#     f.write("rM.append("+str(r_m)+")\n")
# 
#     f.write("aADD.append("+str(a_add)+")\n")
#     f.write("aADHD.append("+str(a_adhd)+")\n")
#     f.write("aAD.append("+str(a_ad)+")\n")
#     f.write("aCP.append("+str(a_cp)+")\n")
#     f.write("abvFTD.append("+str(a_bvftd)+")\n")
#     f.write("aPD.append("+str(a_pd)+")\n")
#     f.write("aM.append("+str(a_m)+")\n")

    f.close()
    
    
    np.save("./output/r_QL_MDP_"+str(scenario_id)+".npy",r_ql,allow_pickle=True)
    np.save("./output/r_DQL_MDP_"+str(scenario_id)+".npy",r_dql,allow_pickle=True)
    np.save("./output/r_SQL_MDP_"+str(scenario_id)+".npy",r_sql,allow_pickle=True)
    np.save("./output/r_PQL_MDP_"+str(scenario_id)+".npy",r_pql,allow_pickle=True)
    np.save("./output/r_NQL_MDP_"+str(scenario_id)+".npy",r_nql,allow_pickle=True)
    np.save("./output/r_EDQL_MDP_"+str(scenario_id)+".npy",r_edql,allow_pickle=True)
    np.save("./output/r_ESQL_MDP_"+str(scenario_id)+".npy",r_esql,allow_pickle=True)
    np.save("./output/r_SARSA_MDP_"+str(scenario_id)+".npy",r_sarsa,allow_pickle=True)
    
    np.save("./output/a_QL_MDP_"+str(scenario_id)+".npy",a_ql,allow_pickle=True)
    np.save("./output/a_DQL_MDP_"+str(scenario_id)+".npy",a_dql,allow_pickle=True)
    np.save("./output/a_SQL_MDP_"+str(scenario_id)+".npy",a_sql,allow_pickle=True)
    np.save("./output/a_PQL_MDP_"+str(scenario_id)+".npy",a_pql,allow_pickle=True)
    np.save("./output/a_NQL_MDP_"+str(scenario_id)+".npy",a_nql,allow_pickle=True)
    np.save("./output/a_EDQL_MDP_"+str(scenario_id)+".npy",a_edql,allow_pickle=True)
    np.save("./output/a_ESQL_MDP_"+str(scenario_id)+".npy",a_esql,allow_pickle=True)
    np.save("./output/a_SARSA_MDP_"+str(scenario_id)+".npy",a_sarsa,allow_pickle=True)
    
    np.save("./output/r_ADD_MDP_"+str(scenario_id)+".npy",r_add,allow_pickle=True)
    np.save("./output/r_ADHD_MDP_"+str(scenario_id)+".npy",r_adhd,allow_pickle=True)
    np.save("./output/r_AD_MDP_"+str(scenario_id)+".npy",r_ad,allow_pickle=True)
    np.save("./output/r_CP_MDP_"+str(scenario_id)+".npy",r_cp,allow_pickle=True)
    np.save("./output/r_bvFTD_MDP_"+str(scenario_id)+".npy",r_bvftd,allow_pickle=True)
    np.save("./output/r_PD_MDP_"+str(scenario_id)+".npy",r_pd,allow_pickle=True)
    np.save("./output/r_M_MDP_"+str(scenario_id)+".npy",r_m,allow_pickle=True)

    np.save("./output/a_ADD_MDP_"+str(scenario_id)+".npy",a_add,allow_pickle=True)
    np.save("./output/a_ADHD_MDP_"+str(scenario_id)+".npy",a_adhd,allow_pickle=True)
    np.save("./output/a_AD_MDP_"+str(scenario_id)+".npy",a_ad,allow_pickle=True)
    np.save("./output/a_CP_MDP_"+str(scenario_id)+".npy",a_cp,allow_pickle=True)
    np.save("./output/a_bvFTD_MDP_"+str(scenario_id)+".npy",a_bvftd,allow_pickle=True)
    np.save("./output/a_PD_MDP_"+str(scenario_id)+".npy",a_pd,allow_pickle=True)
    np.save("./output/a_M_MDP_"+str(scenario_id)+".npy",a_m,allow_pickle=True)

