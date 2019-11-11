from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

# these are the control variables, change them to customize the execution of this program
cntExperiments= 200 # number of experiments to run, large number means longer execution time
MAX_ITER = 500      # number of episodes per experiment, large number means longer execution time
ACTIONS_FOR_B = 50  #number of actions at state B
ACTIONS_FOR_C = 50  #number of actions at state B

#identify the states

# if in IGT, change isIGT to True
isIGT = True

if isIGT:
    STATE_I = 0
    STATE_A = 1
    STATE_B = 2
    STATE_C = 3
    STATE_D = 4
    STATE_E = 5
else:
    STATE_I = 4
    STATE_A = 0
    STATE_B = 1
    STATE_C = 2
    STATE_D = 3
    STATE_E = 4

#identify the actions
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_A = 0
ACTION_B = 1
ACTION_C = 2
ACTION_D = 3

#identify the rewards
REF_Q1 = 0 # Q1 or Q+
REF_Q2 = 1 # Q2 or Q-
REF_Q3 = 3 # Q3 or Q++
REF_Q4 = 4 # Q4 or Q--

# map actions to states
actionsPerState = {}

if isIGT:
    actionsPerState[STATE_I] = [ACTION_A,ACTION_B,ACTION_C,ACTION_D]
    actionsPerState[STATE_A] = []
    actionsPerState[STATE_B] = []
    actionsPerState[STATE_C] = []
    actionsPerState[STATE_D] = []
else:
    actionsPerState = {}
    actionsPerState[STATE_A] = [ACTION_LEFT, ACTION_RIGHT]
    actionsPerState[STATE_B] = [i for i in range(ACTIONS_FOR_B)]
    actionsPerState[STATE_C] = [i for i in range(ACTIONS_FOR_C)]
    # actionsPerState[STATE_C] = [ACTION_RIGHT]
    actionsPerState[STATE_D] = [ACTION_LEFT]
    actionsPerState[STATE_E] = [ACTION_RIGHT]


# init Q values
Q1={}
Q2={}

GAMMA = 0.95

# reset the variables, to be called on each experiment
def reset():
    Q1[STATE_A] = {}
    Q1[STATE_A][ACTION_LEFT] = 0
    Q1[STATE_A][ACTION_RIGHT] = 0

    Q1[STATE_B] = {}
    Q1[STATE_C] = {}

    Q1[STATE_E] = {}
    Q1[STATE_E][ACTION_LEFT] = 0
    Q1[STATE_E][ACTION_RIGHT] = 0
    
    Q1[STATE_D] = {}
    Q1[STATE_D][ACTION_LEFT] = 0
    Q1[STATE_D][ACTION_RIGHT] = 0

    Q2[STATE_A] = {}
    Q2[STATE_A][ACTION_LEFT] = 0
    Q2[STATE_A][ACTION_RIGHT] = 0

    Q2[STATE_B] = {}
    Q2[STATE_C] = {}

    Q2[STATE_E] = {}
    Q2[STATE_E][ACTION_LEFT] = 0
    Q2[STATE_E][ACTION_RIGHT] = 0
    
    Q2[STATE_D] = {}
    Q2[STATE_D][ACTION_LEFT] = 0
    Q2[STATE_D][ACTION_RIGHT] = 0
    
    for i in range(ACTIONS_FOR_B):
        Q1[STATE_B][i] = 0
        Q2[STATE_B][i] = 0
    for i in range(ACTIONS_FOR_C):
        Q1[STATE_C][i] = 0
        Q2[STATE_C][i] = 0
        
# reset the variables, to be called on each experiment
def resetIGT():
    
    STATE_I = 0
    STATE_A = 1
    STATE_B = 2
    STATE_C = 3
    STATE_D = 4
    STATE_E = 5

    actionsPerState[STATE_I] = [ACTION_A,ACTION_B,ACTION_C,ACTION_D]
    actionsPerState[STATE_A] = []
    actionsPerState[STATE_B] = []
    actionsPerState[STATE_C] = []
    actionsPerState[STATE_D] = []
    
    Q1[STATE_I] = {}
    Q1[STATE_I][ACTION_A] = 0
    Q1[STATE_I][ACTION_B] = 0
    Q1[STATE_I][ACTION_C] = 0
    Q1[STATE_I][ACTION_D] = 0

    Q1[STATE_A] = {}
    Q1[STATE_A][ACTION_A] = 0
    Q1[STATE_A][ACTION_B] = 0
    Q1[STATE_A][ACTION_C] = 0
    Q1[STATE_A][ACTION_D] = 0
    
    Q1[STATE_B] = {}
    Q1[STATE_B][ACTION_A] = 0
    Q1[STATE_B][ACTION_B] = 0
    Q1[STATE_B][ACTION_C] = 0
    Q1[STATE_B][ACTION_D] = 0

    Q1[STATE_C] = {}
    Q1[STATE_C][ACTION_A] = 0
    Q1[STATE_C][ACTION_B] = 0
    Q1[STATE_C][ACTION_C] = 0
    Q1[STATE_C][ACTION_D] = 0
    
    Q1[STATE_D] = {}
    Q1[STATE_D][ACTION_A] = 0
    Q1[STATE_D][ACTION_B] = 0
    Q1[STATE_D][ACTION_C] = 0
    Q1[STATE_D][ACTION_D] = 0 
    
    Q2[STATE_I] = {}
    Q2[STATE_I][ACTION_A] = 0
    Q2[STATE_I][ACTION_B] = 0
    Q2[STATE_I][ACTION_C] = 0
    Q2[STATE_I][ACTION_D] = 0

    Q2[STATE_A] = {}
    Q2[STATE_A][ACTION_A] = 0
    Q2[STATE_A][ACTION_B] = 0
    Q2[STATE_A][ACTION_C] = 0
    Q2[STATE_A][ACTION_D] = 0
    
    Q2[STATE_B] = {}
    Q2[STATE_B][ACTION_A] = 0
    Q2[STATE_B][ACTION_B] = 0
    Q2[STATE_B][ACTION_C] = 0
    Q2[STATE_B][ACTION_D] = 0

    Q2[STATE_C] = {}
    Q2[STATE_C][ACTION_A] = 0
    Q2[STATE_C][ACTION_B] = 0
    Q2[STATE_C][ACTION_C] = 0
    Q2[STATE_C][ACTION_D] = 0
    
    Q2[STATE_D] = {}
    Q2[STATE_D][ACTION_A] = 0
    Q2[STATE_D][ACTION_B] = 0
    Q2[STATE_D][ACTION_C] = 0
    Q2[STATE_D][ACTION_D] = 0 
          
# epsilon greedy action
# it return action a 1-epsilon times
# and a random action epsilon times
def random_action(s, a, eps=.1):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(actionsPerState[s])

# move from state s using action a
# it returns the reward and the new state
def move(s, a,reward_from_B,reward_from_C):
    if(s==STATE_A):
        if(a == ACTION_LEFT): 
            return 0, STATE_B
        else: 
            return 0, STATE_C
    if s==STATE_B:
        return reward_from_B(), STATE_D
    if s==STATE_C:
        return reward_from_C(), STATE_E
    return 0, s

def moveIGT(s,a,reward_from_A,reward_from_B,reward_from_C,reward_from_D,N):
    
    STATE_I = 0
    STATE_A = 1
    STATE_B = 2
    STATE_C = 3
    STATE_D = 4
    STATE_E = 5

    actionsPerState[STATE_I] = [ACTION_A,ACTION_B,ACTION_C,ACTION_D]
    actionsPerState[STATE_A] = []
    actionsPerState[STATE_B] = []
    actionsPerState[STATE_C] = []
    actionsPerState[STATE_D] = []
    
    if(s==STATE_I):
        if(a == ACTION_A): 
            return reward_from_A(N[STATE_A]), STATE_A
        elif(a == ACTION_B): 
            return reward_from_B(N[STATE_B]), STATE_B
        elif(a == ACTION_C): 
            return reward_from_C(N[STATE_C]), STATE_C
        elif(a == ACTION_D): 
            return reward_from_D(N[STATE_D]), STATE_D
    return 0, s

# returns the action that makes the max Q value, as well as the max Q value
def maxQA(q, s, isIGT=isIGT,isStart=False):
    max=-9999
    sa = 0
#     print(q,s)
    for k in q[s]:
        if(q[s][k] > max):
            max = q[s][k]
            sa = k
        elif(q[s][k] == max):
            if(np.random.random() < 0.5):
                sa = k
    if isStart and isIGT:
        sa = np.random.randint(4)
        max = 0
    return sa, max
     

# return true if this is a terminal state
def isTerminal(s):
#     return s == STATE_C or s == STATE_D
    return s == STATE_E or s == STATE_D

# return true if this is a terminal state
def isTerminalIGT(s):
    
    STATE_I = 0
    STATE_A = 1
    STATE_B = 2
    STATE_C = 3
    STATE_D = 4
    STATE_E = 5

    actionsPerState[STATE_I] = [ACTION_A,ACTION_B,ACTION_C,ACTION_D]
    actionsPerState[STATE_A] = []
    actionsPerState[STATE_B] = []
    actionsPerState[STATE_C] = []
    actionsPerState[STATE_D] = []
#     return s == STATE_C or s == STATE_D
    return s == STATE_A or s == STATE_B or s == STATE_C or s == STATE_D

# do the experiment by running MAX_ITER episodes and fill the restults in the episods parameter
def experiment(episods, algorithm,reward_from_B,reward_from_C):
    reset()
    #contains the number of times left action is chosen at A
    ALeft = 0

    # contains the number of visits for each state
    N={}
    N[STATE_A] = 1
    N[STATE_B] = 1
    N[STATE_C] = 1
    N[STATE_D] = 1
    N[STATE_E] = 1
    
    M={}
    M[REF_Q1] = 0
    M[REF_Q2] = 0
    M[REF_Q3] = 0
    M[REF_Q4] = 0

    # contains the number of visits for each state and action
    NSA = {}
    
    t = 0
    
    reward = None
    pos_reward = None
    neg_reward = None
    actions = None

    # loop for MAX_ITER episods
    for i in range(MAX_ITER):

        s = STATE_A
        gameover = False

        # use greedy for the action at STATE A
        a = selectInitialAction(algorithm, s)

        #loop until game is over, this will be ONE episode
        while not gameover:
            actions = a
            
            # record learning steps
            t += 1

            # apply epsilon greedy selection (including for action chosen at STATE A)
            a = random_action(s, a, 0.05)

            #update the number of visits for state s
            N[s] += 1

            # if left action is chosen at state A, increment the counter
            if (s == STATE_A and a == ACTION_LEFT):
                ALeft += 1

            #move to the next state and get the reward
            r, nxt_s = move(s, a,reward_from_B,reward_from_C)
            reward = r
            if r > 0: 
                pos_reward = r
                neg_reward = 0
            else:
                pos_reward = 0
                neg_reward = r                
#             reward.append(r)

            #update the number of visits per state and action
            if not s in NSA:
                NSA[s] = {}
            if not a in NSA[s]:
                NSA[s][a] = 0
            NSA[s][a] += 1

            #compute alpha
            alpha = 1 / np.power(NSA[s][a], .8)

            #update the Q values and get the best action for the next state
            nxt_a, M = updateQValues(algorithm, s, a, r, nxt_s, alpha, t, M)

            #if next state is terminal then mark as gameover (end of episode)
            gameover = isTerminal(nxt_s)

            s = nxt_s
            a = nxt_a

        #update stats for each episode
        if not (i in episods):
            episods[i] = {}
            episods[i]["count"] = 0
            episods[i]["Q1(A)l"] = 0
            episods[i]["Q2(A)l"] = 0
            episods[i]["Q1(A)r"] = 0
            episods[i]["Q2(A)r"] = 0
        episods[i]["count"] = ALeft
        episods[i]["percent"] = ALeft / (i+1)
        episods[i]["reward"] = reward
        episods[i]["pos_reward"] = pos_reward
        episods[i]["neg_reward"] = neg_reward
        episods[i]["actions"] = actions
#         episods[i]["cumreward"] = sum(reward)
        episods[i]["Q1(A)l"] = (episods[i]["Q1(A)l"] * i + Q1[STATE_A][ACTION_LEFT])/(i+1)
        episods[i]["Q2(A)l"] = (episods[i]["Q2(A)l"] * i + Q2[STATE_A][ACTION_LEFT])/(i+1)
        episods[i]["Q1(A)r"] = (episods[i]["Q1(A)r"] * i + Q1[STATE_A][ACTION_RIGHT])/(i+1)
        episods[i]["Q2(A)r"] = (episods[i]["Q2(A)r"] * i + Q2[STATE_A][ACTION_RIGHT])/(i+1)
        

# do the experiment by running MAX_ITER episodes and fill the restults in the episods parameter
def experimentIGT(episods, algorithm,reward_from_A,reward_from_B,reward_from_C,reward_from_D,p1=1,p2=1,n1=1,n2=1):
    resetIGT()
    
    STATE_I = 0
    STATE_A = 1
    STATE_B = 2
    STATE_C = 3
    STATE_D = 4
    STATE_E = 5

    actionsPerState[STATE_I] = [ACTION_A,ACTION_B,ACTION_C,ACTION_D]
    actionsPerState[STATE_A] = []
    actionsPerState[STATE_B] = []
    actionsPerState[STATE_C] = []
    actionsPerState[STATE_D] = []
    
    #contains the number of times left action is chosen at initial state I
    ILeft = 0
    
    # contains the number of visits for each state
    N={}
    N[STATE_I] = 1
    N[STATE_A] = 1
    N[STATE_B] = 1
    N[STATE_C] = 1
    N[STATE_D] = 1
    
    M={}
    M[REF_Q1] = 0
    M[REF_Q2] = 0
    M[REF_Q3] = 0
    M[REF_Q4] = 0

    # contains the number of visits for each state and action
    NSA = {}
    
    t = 0
    
    reward = None
    pos_reward = None
    neg_reward = None
    actions = None
    # loop for MAX_ITER episods
    for i in range(MAX_ITER):

        s = STATE_I
        gameover = False

        # use greedy for the action at STATE A
        if i == 0:
            a = selectInitialActionIGT(algorithm,s,p1,p2,n1,n2,veryFirst=True)
        else:
            a = selectInitialActionIGT(algorithm,s,p1,p2,n1,n2)

        #loop until game is over, this will be ONE episode
        while not gameover:
            actions = a
            
            # record learning steps
            t += 1

            # apply epsilon greedy selection (including for action chosen at STATE A)
            a = random_action(s, a, 0.05)

            #update the number of visits for state s
            N[s] += 1

            # if left action is chosen at state A, increment the counter
            if (s == STATE_I and a == ACTION_A) or (s == STATE_I and a == ACTION_B):
                ILeft += 1

            #move to the next state and get the reward
            [pr,nr], nxt_s = moveIGT(s, a,reward_from_A,reward_from_B,reward_from_C,reward_from_D,N)
            r = pr+nr
#             reward.append(r)
#             pos_reward.append(pr)
#             neg_reward.append(nr)
            reward = r
            pos_reward = pr
            neg_reward = nr

            #update the number of visits per state and action
            if not s in NSA:
                NSA[s] = {}
            if not a in NSA[s]:
                NSA[s][a] = 0
            NSA[s][a] += 1

            #compute alpha
            alpha = 1 / np.power(NSA[s][a], .8)

            #update the Q values and get the best action for the next state
            nxt_a, M = updateQValuesIGT(algorithm, s, a, r,pr,nr, nxt_s, alpha, t, M,p1,p2,n1,n2)

            #if next state is terminal then mark as gameover (end of episode)
            gameover = isTerminalIGT(nxt_s)

            if gameover: N[nxt_s] += 1
            s = nxt_s
            a = nxt_a

        #update stats for each episode
        if not (i in episods):
            episods[i] = {}
            episods[i]["count"] = 0
            episods[i]["Q1(I)a"] = 0
            episods[i]["Q2(I)a"] = 0
            episods[i]["Q1(I)b"] = 0
            episods[i]["Q2(I)b"] = 0
            episods[i]["Q1(I)c"] = 0
            episods[i]["Q2(I)c"] = 0
            episods[i]["Q1(I)d"] = 0
            episods[i]["Q2(I)d"] = 0
        episods[i]["count"] = ILeft
        episods[i]["percent"] = ILeft / (i+1)
        episods[i]["reward"] = reward
        episods[i]["pos_reward"] = pos_reward
        episods[i]["neg_reward"] = neg_reward
        episods[i]["actions"] = actions
#         episods[i]["cumreward"] = sum(reward)
        episods[i]["Q1(I)a"] = ((episods[i]["Q1(I)a"] * i) + Q1[STATE_I][ACTION_A])/(i+1)
        episods[i]["Q2(I)a"] = ((episods[i]["Q2(I)a"] * i) + Q2[STATE_I][ACTION_A])/(i+1)
        episods[i]["Q1(I)b"] = ((episods[i]["Q1(I)b"] * i) + Q1[STATE_I][ACTION_B])/(i+1)
        episods[i]["Q2(I)b"] = ((episods[i]["Q2(I)b"] * i) + Q2[STATE_I][ACTION_B])/(i+1)
        episods[i]["Q1(I)c"] = ((episods[i]["Q1(I)c"] * i) + Q1[STATE_I][ACTION_C])/(i+1)
        episods[i]["Q2(I)c"] = ((episods[i]["Q2(I)c"] * i) + Q2[STATE_I][ACTION_C])/(i+1)
        episods[i]["Q1(I)d"] = ((episods[i]["Q1(I)d"] * i) + Q1[STATE_I][ACTION_D])/(i+1)
        episods[i]["Q2(I)d"] = ((episods[i]["Q2(I)d"] * i) + Q2[STATE_I][ACTION_D])/(i+1)
        
        

        
# run the learning
def runLearning(algorithm, report, experimentsCount,reward_from_B,reward_from_C,report_full=None):
    #run batch of experiments
    count = np.ndarray((experimentsCount,MAX_ITER))
    percent = np.ndarray((experimentsCount,MAX_ITER))
    Q1Al = np.ndarray((experimentsCount,MAX_ITER))
    Q2Al = np.ndarray((experimentsCount,MAX_ITER))
    Q1Ar = np.ndarray((experimentsCount,MAX_ITER))
    Q2Ar = np.ndarray((experimentsCount,MAX_ITER))
    cumreward = np.ndarray((experimentsCount,MAX_ITER))
    reward = np.ndarray((experimentsCount,MAX_ITER))
    pos_reward = np.ndarray((experimentsCount,MAX_ITER))
    neg_reward = np.ndarray((experimentsCount,MAX_ITER))
    actions = np.ndarray((experimentsCount,MAX_ITER))
    
    for k in range(experimentsCount):
        tmp = {}
        experiment(tmp, algorithm,reward_from_B,reward_from_C)
        #aggregate every experiment result into the final report
        for i in range(MAX_ITER):

            count[k,i] = tmp[i]["count"]
            percent[k,i] = 100*tmp[i]["count"] / (i+1)
            Q1Al[k,i] = tmp[i]["Q1(A)l"]
            Q2Al[k,i] = tmp[i]["Q2(A)l"]
            Q1Ar[k,i] = tmp[i]["Q1(A)r"]
            Q2Ar[k,i] = tmp[i]["Q2(A)r"]
#             cumreward[k,i] = tmp[i]["cumreward"]
            reward[k,i] = tmp[i]["reward"]
            pos_reward[k,i] = tmp[i]["pos_reward"]
            neg_reward[k,i] = tmp[i]["neg_reward"]
            actions[k,i] = tmp[i]["actions"]
            
    report["count"] = count
    report["percent"] = percent
    report["Q1(A)l"] = Q1Al
    report["Q2(A)l"] = Q2Al
    report["Q1(A)r"] = Q1Ar
    report["Q2(A)r"] = Q2Ar
#     report["cumreward"] = cumreward
    report["reward"] = reward
    report["pos_reward"] = pos_reward
    report["neg_reward"] = neg_reward
    report["actions"] = actions

# run the learning
def runIGT(algorithm, report, experimentsCount,reward_from_A,reward_from_B,reward_from_C,reward_from_D,p1=1,p2=1,n1=1,n2=1,report_full=None):
    count = np.ndarray((experimentsCount,MAX_ITER))
    percent = np.ndarray((experimentsCount,MAX_ITER))
    Q1Ia = np.ndarray((experimentsCount,MAX_ITER))
    Q2Ia = np.ndarray((experimentsCount,MAX_ITER))
    Q1Ib = np.ndarray((experimentsCount,MAX_ITER))
    Q2Ib = np.ndarray((experimentsCount,MAX_ITER))
    Q1Ic = np.ndarray((experimentsCount,MAX_ITER))
    Q2Ic = np.ndarray((experimentsCount,MAX_ITER))
    Q1Id = np.ndarray((experimentsCount,MAX_ITER))
    Q2Id = np.ndarray((experimentsCount,MAX_ITER))
    cumreward = np.ndarray((experimentsCount,MAX_ITER))
    reward = np.ndarray((experimentsCount,MAX_ITER))
    pos_reward = np.ndarray((experimentsCount,MAX_ITER))
    neg_reward = np.ndarray((experimentsCount,MAX_ITER))
    actions = np.ndarray((experimentsCount,MAX_ITER))

    #run batch of experiments
    for k in range(experimentsCount):
        tmp = {}
        experimentIGT(tmp,algorithm,reward_from_A,reward_from_B,reward_from_C,reward_from_D,p1,p2,n1,n2)
        #aggregate every experiment result into the final report
        for i in range(MAX_ITER):

            count[k,i] = tmp[i]["count"]
            percent[k,i] = 100*tmp[i]["count"] / (i+1)
            Q1Id[k,i] = tmp[i]["Q1(I)a"]
            Q2Ia[k,i] = tmp[i]["Q2(I)a"]
            Q1Ib[k,i] = tmp[i]["Q1(I)b"]
            Q2Ib[k,i] = tmp[i]["Q2(I)b"]
            Q1Ic[k,i] = tmp[i]["Q1(I)c"]
            Q2Ic[k,i] = tmp[i]["Q2(I)c"]
            Q1Id[k,i] = tmp[i]["Q1(I)d"]
            Q2Id[k,i] = tmp[i]["Q2(I)d"]
#             cumreward[k,i] = tmp[i]["cumreward"]
            reward[k,i] = tmp[i]["reward"]
            pos_reward[k,i] = tmp[i]["pos_reward"]
            neg_reward[k,i] = tmp[i]["neg_reward"]
            actions[k,i] = tmp[i]["actions"]

    report["count"] = count
    report["percent"] = percent
    report["Q1(I)a"] = Q1Ia
    report["Q2(I)a"] = Q2Ia
    report["Q1(I)b"] = Q1Ib
    report["Q2(I)b"] = Q2Ib
    report["Q1(I)c"] = Q1Ic
    report["Q2(I)c"] = Q2Ic
    report["Q1(I)d"] = Q1Id
    report["Q2(I)d"] = Q2Id
    report["cumreward"] = cumreward
    report["reward"] = reward
    report["pos_reward"] = pos_reward
    report["neg_reward"] = neg_reward
    report["actions"] = actions
    

# select the initial action at state A, it uses greedy method
# it takes into the mode doubleQLearning or not
def selectInitialAction(algorithm, startState):
    if algorithm == 'SARSA':
        a, _ = maxQA(Q1, startState)
        
    if algorithm == 'Q-Learning':
        a, _ = maxQA(Q1, startState)
    
    if algorithm == 'Positive Q-Learning':
        a, _ = maxQA(Q1, startState)
        
    if algorithm == 'Negative Q-Learning':
        a, _ = maxQA(Q2, startState)
        
    if algorithm == 'Double Q-Learning':
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = Q1[STATE_A][ACTION_LEFT] + Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = Q1[STATE_A][ACTION_RIGHT] + Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)
        
    if algorithm == 'Split Q-Learning':
        p1,p2,pw,n1,n2,nw = 1,1,1,1,1,1
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = pw*Q1[STATE_A][ACTION_LEFT] + nw*Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = pw*Q1[STATE_A][ACTION_RIGHT] + nw*Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)
                
    if algorithm == 'Exponential Double Q-Learning':
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = Q1[STATE_A][ACTION_LEFT] + Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = Q1[STATE_A][ACTION_RIGHT] + Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)
    
    if algorithm == 'Exponential Split Q-Learning':
        p1,p2,pw,n1,n2,nw = 1,1,1,1,1,1
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = pw*Q1[STATE_A][ACTION_LEFT] + nw*Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = pw*Q1[STATE_A][ACTION_RIGHT] + nw*Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)
        
    if algorithm == 'ADD':
        pw,nw = 1,1
#         p1,pw,n1,nw = np.random.normal(1,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = pw*Q1[STATE_A][ACTION_LEFT] + nw*Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = pw*Q1[STATE_A][ACTION_RIGHT] + nw*Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)

    if algorithm == 'ADHD':
        pw,nw = 1,1
#         p1,pw,n1,nw = np.random.normal(0.2,0.1),np.random.normal(1,0.1),np.random.normal(0.2,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = pw*Q1[STATE_A][ACTION_LEFT] + nw*Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = pw*Q1[STATE_A][ACTION_RIGHT] + nw*Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)

    if algorithm == 'AD':
        pw,nw = 1,1
#         p1,pw,n1,nw = np.random.normal(0.1,0.1),np.random.normal(1,0.1),np.random.normal(0.1,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = pw*Q1[STATE_A][ACTION_LEFT] + nw*Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = pw*Q1[STATE_A][ACTION_RIGHT] + nw*Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)

    if algorithm == 'CP':
        pw,nw = 1,1
#         p1,pw,n1,nw = np.random.normal(0.5,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = pw*Q1[STATE_A][ACTION_LEFT] + nw*Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = pw*Q1[STATE_A][ACTION_RIGHT] + nw*Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)

    if algorithm == 'bvFTD':
        pw,nw = 1,1
#         p1,pw,n1,nw = np.random.normal(0.5,0.1),np.random.normal(100,10),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = pw*Q1[STATE_A][ACTION_LEFT] + nw*Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = pw*Q1[STATE_A][ACTION_RIGHT] + nw*Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)

    if algorithm == 'PD':
        pw,nw = 1,1
#         p1,pw,n1,nw = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(100,10)
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = pw*Q1[STATE_A][ACTION_LEFT] + nw*Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = pw*Q1[STATE_A][ACTION_RIGHT] + nw*Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)

    if algorithm == 'M':
        pw,nw = 1,1
#         p1,pw,n1,nw = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = pw*Q1[STATE_A][ACTION_LEFT] + nw*Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = pw*Q1[STATE_A][ACTION_RIGHT] + nw*Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)

    return a


# select the initial action at state I, it uses greedy method
# it takes into the mode doubleQLearning or not
def selectInitialActionIGT(algorithm, startState,p1,p2,n1,n2,veryFirst=False):
    
    STATE_I = 0
    STATE_A = 1
    STATE_B = 2
    STATE_C = 3
    STATE_D = 4
    STATE_E = 5

    actionsPerState[STATE_I] = [ACTION_A,ACTION_B,ACTION_C,ACTION_D]
    actionsPerState[STATE_A] = []
    actionsPerState[STATE_B] = []
    actionsPerState[STATE_C] = []
    actionsPerState[STATE_D] = []
    
    if algorithm == 'SARSA':
        a, _ = maxQA(Q1, startState,isStart=veryFirst)
            
    if algorithm == 'Q-Learning':
        a, _ = maxQA(Q1, startState,isStart=veryFirst)
    
    if algorithm == 'Positive Q-Learning':
        a, _ = maxQA(Q1, startState,isStart=veryFirst)
        
    if algorithm == 'Negative Q-Learning':
        a, _ = maxQA(Q2, startState,isStart=veryFirst)
        
    if algorithm == 'Double Q-Learning':
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = Q1[STATE_I][ACTION_A] + Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = Q1[STATE_I][ACTION_B] + Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = Q1[STATE_I][ACTION_C] + Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = Q1[STATE_I][ACTION_D] + Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)
        
    if algorithm == 'Split Q-Learning':
        p1,pw,n1,nw = p1,p2,n1,n2
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = pw*Q1[STATE_I][ACTION_A] + nw*Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = pw*Q1[STATE_I][ACTION_B] + nw*Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = pw*Q1[STATE_I][ACTION_C] + nw*Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = pw*Q1[STATE_I][ACTION_D] + nw*Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)
        
    if algorithm == 'Exponential Double Q-Learning':
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = Q1[STATE_I][ACTION_A] + Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = Q1[STATE_I][ACTION_B] + Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = Q1[STATE_I][ACTION_C] + Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = Q1[STATE_I][ACTION_D] + Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)
    
    if algorithm == 'Exponential Split Q-Learning':
        p1,pw,n1,nw = 1,1,1,1
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = pw*Q1[STATE_I][ACTION_A] + nw*Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = pw*Q1[STATE_I][ACTION_B] + nw*Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = pw*Q1[STATE_I][ACTION_C] + nw*Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = pw*Q1[STATE_I][ACTION_D] + nw*Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)
        
    if algorithm == 'ADD':
        p1,pw,n1,nw = np.random.normal(1,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = pw*Q1[STATE_I][ACTION_A] + nw*Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = pw*Q1[STATE_I][ACTION_B] + nw*Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = pw*Q1[STATE_I][ACTION_C] + nw*Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = pw*Q1[STATE_I][ACTION_D] + nw*Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)

    if algorithm == 'ADHD':
        p1,pw,n1,nw = np.random.normal(0.2,0.1),np.random.normal(1,0.1),np.random.normal(0.2,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = pw*Q1[STATE_I][ACTION_A] + nw*Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = pw*Q1[STATE_I][ACTION_B] + nw*Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = pw*Q1[STATE_I][ACTION_C] + nw*Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = pw*Q1[STATE_I][ACTION_D] + nw*Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)

    if algorithm == 'AD':
        p1,pw,n1,nw = np.random.normal(0.1,0.1),np.random.normal(1,0.1),np.random.normal(0.1,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = pw*Q1[STATE_I][ACTION_A] + nw*Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = pw*Q1[STATE_I][ACTION_B] + nw*Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = pw*Q1[STATE_I][ACTION_C] + nw*Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = pw*Q1[STATE_I][ACTION_D] + nw*Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)

    if algorithm == 'CP':
        p1,pw,n1,nw = np.random.normal(0.5,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = pw*Q1[STATE_I][ACTION_A] + nw*Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = pw*Q1[STATE_I][ACTION_B] + nw*Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = pw*Q1[STATE_I][ACTION_C] + nw*Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = pw*Q1[STATE_I][ACTION_D] + nw*Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)

    if algorithm == 'bvFTD':
        p1,pw,n1,nw = np.random.normal(0.5,0.1),np.random.normal(100,10),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = pw*Q1[STATE_I][ACTION_A] + nw*Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = pw*Q1[STATE_I][ACTION_B] + nw*Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = pw*Q1[STATE_I][ACTION_C] + nw*Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = pw*Q1[STATE_I][ACTION_D] + nw*Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)

    if algorithm == 'PD':
        p1,pw,n1,nw = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(100,10)
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = pw*Q1[STATE_I][ACTION_A] + nw*Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = pw*Q1[STATE_I][ACTION_B] + nw*Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = pw*Q1[STATE_I][ACTION_C] + nw*Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = pw*Q1[STATE_I][ACTION_D] + nw*Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)

    if algorithm == 'M':
        p1,pw,n1,nw = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        QS = {}
        QS[STATE_I] = {}
        QS[STATE_I][ACTION_A] = pw*Q1[STATE_I][ACTION_A] + nw*Q2[STATE_I][ACTION_A]
        QS[STATE_I][ACTION_B] = pw*Q1[STATE_I][ACTION_B] + nw*Q2[STATE_I][ACTION_B]
        QS[STATE_I][ACTION_C] = pw*Q1[STATE_I][ACTION_C] + nw*Q2[STATE_I][ACTION_C]
        QS[STATE_I][ACTION_D] = pw*Q1[STATE_I][ACTION_D] + nw*Q2[STATE_I][ACTION_D]
        a, _ = maxQA(QS, startState,isStart=veryFirst)

    return a


#update Q values depending on whether the mode  is doubleQLearning or not
def updateQValues(algorithm, s, a, r, nxt_s, alpha, t, M):

    if algorithm == 'SARSA':
        nxt_a, maxq = maxQA(Q1, nxt_s)
        Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        
    if algorithm == 'Q-Learning':
        nxt_a, maxq = maxQA(Q1, nxt_s)
        Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * maxq - Q1[s][a])
    
    if algorithm == 'Double Q-Learning':
        p = np.random.random()
        if (p < .5):
            nxt_a, maxq = maxQA(Q1, nxt_s)
            Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * Q2[nxt_s][nxt_a] - Q1[s][a])
        else:
            nxt_a, maxq = maxQA(Q2, nxt_s)
            Q2[s][a] = Q2[s][a] + alpha * (r + GAMMA * Q1[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'Split Q-Learning':
        p1,p2,n1,n2 = 1,1,1,1
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        if (r <= 0):
            Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])
           
    if algorithm == 'Exponential Double Q-Learning':
        rho = 1
        p = np.random.random()
        if (p < .5):
            r = r*math.exp(r/rho)
            nxt_a, maxq = maxQA(Q1, nxt_s)
            Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * Q2[nxt_s][nxt_a] - Q1[s][a])
        else:
            nxt_a, maxq = maxQA(Q2, nxt_s)
            r = r*math.exp(-r/rho)
            Q2[s][a] = Q2[s][a] + alpha * (r + GAMMA * Q1[nxt_s][nxt_a] - Q2[s][a])
    
    if algorithm == 'Exponential Split Q-Learning':
        p1,p2,n1,n2 = 1,1,1,1
        rho = 1
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            r = r*math.exp(r/rho)
            Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        if (r <= 0):
            r = r*math.exp(-r/rho)
            Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])
            
    if algorithm == 'Positive Q-Learning':
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
            
    if algorithm == 'Negative Q-Learning':
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r <= 0):
            Q2[s][a] = Q2[s][a] + alpha * (r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])
        
    if algorithm == 'ADD':
        p1,p2,n1,n2 = np.random.normal(1,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        if (r <= 0):
            Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'ADHD':
        p1,p2,n1,n2 = np.random.normal(0.2,0.1),np.random.normal(1,0.1),np.random.normal(0.2,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        if (r <= 0):
            Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'AD':
        p1,p2,n1,n2 = np.random.normal(0.1,0.1),np.random.normal(1,0.1),np.random.normal(0.1,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        if (r <= 0):
            Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'CP':
        p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        if (r <= 0):
            Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'bvFTD':
        p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(100,10),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        if (r <= 0):
            Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'PD':
        p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(100,10)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        if (r <= 0):
            Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'M':
        p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_LEFT] = Q1[nxt_s][ACTION_LEFT] + Q2[nxt_s][ACTION_LEFT]
        Qprime[nxt_s][ACTION_RIGHT] = Q1[nxt_s][ACTION_RIGHT] + Q2[nxt_s][ACTION_RIGHT]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            Q1[s][a] = p1*Q1[s][a] + alpha * (p2*r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        if (r <= 0):
            Q2[s][a] = n1*Q2[s][a] + alpha * (n2*r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    return nxt_a, M


#update Q values depending on whether the mode  is doubleQLearning or not
def updateQValuesIGT(algorithm, s, a, r,pr, nr, nxt_s, alpha, t, M,p1,p2,n1,n2):
    
    STATE_I = 0
    STATE_A = 1
    STATE_B = 2
    STATE_C = 3
    STATE_D = 4
    STATE_E = 5

    actionsPerState[STATE_I] = [ACTION_A,ACTION_B,ACTION_C,ACTION_D]
    actionsPerState[STATE_A] = []
    actionsPerState[STATE_B] = []
    actionsPerState[STATE_C] = []
    actionsPerState[STATE_D] = []

    
    if algorithm == 'SARSA':
        nxt_a, maxq = maxQA(Q1, nxt_s)
        Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
            
    if algorithm == 'Q-Learning':
        nxt_a, maxq = maxQA(Q1, nxt_s)
        Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * maxq - Q1[s][a])
    
    if algorithm == 'Double Q-Learning':
        p = np.random.random()
        if (p < .5):
            nxt_a, maxq = maxQA(Q1, nxt_s)
            Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * Q2[nxt_s][nxt_a] - Q1[s][a])
        else:
            nxt_a, maxq = maxQA(Q2, nxt_s)
            Q2[s][a] = Q2[s][a] + alpha * (r + GAMMA * Q1[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'Split Q-Learning':
        p1,p2,n1,n2 = 1,1,1,1
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])
           
    if algorithm == 'Exponential Double Q-Learning':
        rho = 1
        p = np.random.random()
        if (p < .5):
            try:
                r = r*math.exp(r/rho)
            except:
                r = r
            nxt_a, maxq = maxQA(Q1, nxt_s)
            Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * Q2[nxt_s][nxt_a] - Q1[s][a])
        else:
            nxt_a, maxq = maxQA(Q2, nxt_s)
            try:
                r = r*math.exp(-r/rho)
            except:
                r = r
            Q2[s][a] = Q2[s][a] + alpha * (r + GAMMA * Q1[nxt_s][nxt_a] - Q2[s][a])
    
    if algorithm == 'Exponential Split Q-Learning':

        p1,p2,n1,n2 = p1,p2,n1,n2
        rho = 1
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        try:
            pr = pr*math.exp(pr/rho)
        except:
            pr = pr
        Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        try:
            nr = nr*math.exp(-nr/rho)
        except:
            nr = nr
        Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])
            
    if algorithm == 'Positive Q-Learning':
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r >= 0):
            Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
            
    if algorithm == 'Negative Q-Learning':
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        if (r <= 0):
            Q2[s][a] = Q2[s][a] + alpha * (r + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])
        
    if algorithm == 'ADD':
        p1,p2,n1,n2 = np.random.normal(1,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'ADHD':
        p1,p2,n1,n2 = np.random.normal(0.2,0.1),np.random.normal(1,0.1),np.random.normal(0.2,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'AD':
        p1,p2,n1,n2 = np.random.normal(0.1,0.1),np.random.normal(1,0.1),np.random.normal(0.1,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'CP':
        p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(1,0.1)        
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'bvFTD':
        p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(100,10),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'PD':
        p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(100,10)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])

    if algorithm == 'M':
        p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        Qprime = {}
        Qprime[nxt_s] = {}
        Qprime[nxt_s][ACTION_A] = Q1[nxt_s][ACTION_A] + Q2[nxt_s][ACTION_A]
        Qprime[nxt_s][ACTION_B] = Q1[nxt_s][ACTION_B] + Q2[nxt_s][ACTION_B]
        Qprime[nxt_s][ACTION_C] = Q1[nxt_s][ACTION_C] + Q2[nxt_s][ACTION_C]
        Qprime[nxt_s][ACTION_D] = Q1[nxt_s][ACTION_D] + Q2[nxt_s][ACTION_D]
        nxt_a, maxq = maxQA(Qprime, nxt_s)
        Q1[s][a] = p1*Q1[s][a] + alpha * (p2*pr + GAMMA * Q1[nxt_s][nxt_a] - Q1[s][a])
        Q2[s][a] = n1*Q2[s][a] + alpha * (n2*nr + GAMMA * Q2[nxt_s][nxt_a] - Q2[s][a])
     
    return nxt_a, M


