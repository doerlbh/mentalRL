from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('should have a scheme arguments. 1-2 for IGT scheme')
    try:
        scheme_id = int(sys.argv[1])
    except:
        sys.exit('Wrong second argument')

    fig_name,reward_from_A,reward_from_B,reward_from_C,reward_from_D = load_IGT(scheme_id)

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

    runIGT('Q-Learning',reportQL,cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('Double Q-Learning',reportDoubleQL,cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('Split Q-Learning',reportSplitQL,cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('Positive Q-Learning',reportPositiveQl,cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('Negative Q-Learning',reportNegativeQl,cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('Exponential Double Q-Learning',reportExponentialDoubleQl, cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('Exponential Split Q-Learning',reportExponentialSplitQl, cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('SARSA',reportSARSA, cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    
    runIGT('ADD',reportADD, cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('ADHD',reportADHD, cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('AD',reportAD, cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('CP',reportCP, cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('bvFTD',reportbvFTD, cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('PD',reportPD, cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)
    runIGT('M',reportM, cntExperiments,reward_from_A,reward_from_B,reward_from_C,reward_from_D)

    # print graphs
    fig,ql,dql,sql,pql,nql,edql,esql,sarsa,r_ql,r_dql,r_sql,r_pql,r_nql,r_edql,r_esql,r_sarsa,a_ql,a_dql,a_sql,a_pql,a_nql,a_edql,a_esql,a_sarsa = drawGraphIGT(reportQL,reportDoubleQL,reportSplitQL,reportPositiveQl,reportNegativeQl,reportExponentialDoubleQl,reportExponentialSplitQl,reportSARSA,reward_from_A,reward_from_B,reward_from_C,reward_from_D,fig_name+'_mental_1.png',draw_SARSA=True)
    fig,add,adhd,ad,cp,bvftd,pd,m,r_add,r_adhd,r_ad,r_cp,r_bvftd,r_pd,r_m,a_add,a_adhd,a_ad,a_cp,a_bvftd,a_pd,a_m = drawGraphIGT_mental(reportADD,reportADHD,reportAD,reportCP,reportbvFTD,reportPD,reportM,reward_from_A,reward_from_B,reward_from_C,reward_from_D,fig_name+'_mental_2.png',plotShortTerm=True)
    
    with open('IGT_mental_wi_'+str(scheme_id)+'.csv', 'wb') as record:
        np.savetxt(record, reportADD["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportADHD["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportAD["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportCP["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportbvFTD["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportPD["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportM["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportQL["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportDoubleQL["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportSplitQL["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportPositiveQl["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportNegativeQl["pos_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportSARSA["pos_reward"], delimiter=",",fmt='%5.2f')

    with open('IGT_mental_lo_'+str(scheme_id)+'.csv', 'wb') as record:
        np.savetxt(record, reportADD["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportADHD["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportAD["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportCP["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportbvFTD["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportPositiveQl["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportM["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportQL["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportDoubleQL["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportSplitQL["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportPositiveQl["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportNegativeQl["neg_reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportSARSA["neg_reward"], delimiter=",",fmt='%5.2f')

    with open('IGT_mental_gn_'+str(scheme_id)+'.csv', 'wb') as record:
        np.savetxt(record, reportADD["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportADHD["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportAD["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportCP["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportbvFTD["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportPD["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportM["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportQL["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportDoubleQL["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportSplitQL["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportPositiveQl["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportNegativeQl["reward"], delimiter=",",fmt='%5.2f')
        np.savetxt(record, reportSARSA["reward"], delimiter=",",fmt='%5.2f')

    f = open("IGT_mental_index.csv", "a")
    for i in np.arange(cntExperiments): f.write("ADD\n")
    for i in np.arange(cntExperiments): f.write("ADHD\n")
    for i in np.arange(cntExperiments): f.write("AD\n")
    for i in np.arange(cntExperiments): f.write("CP\n")
    for i in np.arange(cntExperiments): f.write("bvFTD\n")
    for i in np.arange(cntExperiments): f.write("PD\n")
    for i in np.arange(cntExperiments): f.write("M\n")
    for i in np.arange(cntExperiments): f.write("QL\n")
    for i in np.arange(cntExperiments): f.write("DQL\n")
    for i in np.arange(cntExperiments): f.write("SQL\n")
    for i in np.arange(cntExperiments): f.write("PQL\n")
    for i in np.arange(cntExperiments): f.write("NQL\n")
    for i in np.arange(cntExperiments): f.write("SARSA\n")
    f.close()

    f = open("IGT_results.py", "a")

    f.write("\n")
    f.write("scenario.append("+str(scheme_id)+")\n")
#     f.write("kld.append("+str(kld)+")\n")
#     f.write("sB.append("+str(sB)+")\n")
#     f.write("sC.append("+str(sC)+")\n")
#     f.write("sdiff.append("+str(sdiff)+")\n")
#     f.write("crossentropy.append("+str(crossentropy)+")\n")

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

    np.save("./output/r_QL_IGT_"+str(scheme_id)+".npy",r_ql,allow_pickle=True)
    np.save("./output/r_DQL_IGT_"+str(scheme_id)+".npy",r_dql,allow_pickle=True)
    np.save("./output/r_SQL_IGT_"+str(scheme_id)+".npy",r_sql,allow_pickle=True)
    np.save("./output/r_PQL_IGT_"+str(scheme_id)+".npy",r_pql,allow_pickle=True)
    np.save("./output/r_NQL_IGT_"+str(scheme_id)+".npy",r_nql,allow_pickle=True)
    np.save("./output/r_EDQL_IGT_"+str(scheme_id)+".npy",r_edql,allow_pickle=True)
    np.save("./output/r_ESQL_IGT_"+str(scheme_id)+".npy",r_esql,allow_pickle=True)
    np.save("./output/r_SARSA_IGT_"+str(scheme_id)+".npy",r_sarsa,allow_pickle=True)
    
    np.save("./output/a_QL_IGT_"+str(scheme_id)+".npy",a_ql,allow_pickle=True)
    np.save("./output/a_DQL_IGT_"+str(scheme_id)+".npy",a_dql,allow_pickle=True)
    np.save("./output/a_SQL_IGT_"+str(scheme_id)+".npy",a_sql,allow_pickle=True)
    np.save("./output/a_PQL_IGT_"+str(scheme_id)+".npy",a_pql,allow_pickle=True)
    np.save("./output/a_NQL_IGT_"+str(scheme_id)+".npy",a_nql,allow_pickle=True)
    np.save("./output/a_EDQL_IGT_"+str(scheme_id)+".npy",a_edql,allow_pickle=True)
    np.save("./output/a_ESQL_IGT_"+str(scheme_id)+".npy",a_esql,allow_pickle=True)
    np.save("./output/a_SARSA_IGT_"+str(scheme_id)+".npy",a_sarsa,allow_pickle=True)
    
    np.save("./output/r_ADD_IGT_"+str(scheme_id)+".npy",r_add,allow_pickle=True)
    np.save("./output/r_ADHD_IGT_"+str(scheme_id)+".npy",r_adhd,allow_pickle=True)
    np.save("./output/r_AD_IGT_"+str(scheme_id)+".npy",r_ad,allow_pickle=True)
    np.save("./output/r_CP_IGT_"+str(scheme_id)+".npy",r_cp,allow_pickle=True)
    np.save("./output/r_bvFTD_IGT_"+str(scheme_id)+".npy",r_bvftd,allow_pickle=True)
    np.save("./output/r_PD_IGT_"+str(scheme_id)+".npy",r_pd,allow_pickle=True)
    np.save("./output/r_M_IGT_"+str(scheme_id)+".npy",r_m,allow_pickle=True)

    np.save("./output/a_ADD_IGT_"+str(scheme_id)+".npy",a_add,allow_pickle=True)
    np.save("./output/a_ADHD_IGT_"+str(scheme_id)+".npy",a_adhd,allow_pickle=True)
    np.save("./output/a_AD_IGT_"+str(scheme_id)+".npy",a_ad,allow_pickle=True)
    np.save("./output/a_CP_IGT_"+str(scheme_id)+".npy",a_cp,allow_pickle=True)
    np.save("./output/a_bvFTD_IGT_"+str(scheme_id)+".npy",a_bvftd,allow_pickle=True)
    np.save("./output/a_PD_IGT_"+str(scheme_id)+".npy",a_pd,allow_pickle=True)
    np.save("./output/a_M_IGT_"+str(scheme_id)+".npy",a_m,allow_pickle=True)


