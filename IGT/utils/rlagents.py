from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

# RL Algorithms:
# QL: Q-learning
# DQL: Double Q-learning
# SQL: Split Q-learning
# EDQL: Exponential Double Q-learning
# ESQL: Exponential Split Q-learning
# PQL: Positive Q-learning
# NQL: Negative Q-learning
# SARSA: State–Action–Reward–State–Action
# MP: MaxPain
# ADD: SQL for Addiction
# AD: SQL for Alzheimer's Disease
# ADHD: SQL for Attention-Deficit Hyperactivity Disorder
# PD: SQL for Parkinson's Disorder
# bvFTD: SQL for behavioral variant of FrontoTemporal Dementia
# CP: SQL for Chronic Pain
# M: SQL for Moderate

# Bandits Algorithms:
# TS: Thompson Sampling
# UCB: Upper Confidence Bound (implemented UCB1)
# HBTS: Human Behavior Thompson Sampling (Split Thompson Sampling)
# ETS: Exponential Thompson Sampling
# EHBTS: Exponential Human Behavior Thompson Sampling
# PTS: Positive Thompson Sampling
# NTS: Negative Thompson Sampling
# eGreedy: epsilon Greedy
# EXP3: EXP3
# EXP30: EXP3 with mu set as zero
# bADD: HBTS for Addiction
# bAD: HBTS for Alzheimer's Disease
# bADHD: HBTS for Attention-Deficit Hyperactivity Disorder
# bPD: HBTS for Parkinson's Disorder
# bbvFTD: HBTS for behavioral variant of FrontoTemporal Dementia
# bCP: HBTS for Chronic Pain
# bM: HBTS for Moderate

# Contexutal Bandits Algorithms:
# CTS: Contextual Thompson Sampling
# LinUCB: Upper Confidence Bound (implemented UCB1)
# SCTS: Split Contextual Thompson Sampling
# EXP4: EXP4
# PCTS: Positive Contextual Thompson Sampling
# NCTS: Negative Contextual Thompson Sampling
# cADD: SCTS for Addiction
# cAD: SCTS for Alzheimer's Disease
# cADHD: SCTS for Attention-Deficit Hyperactivity Disorder
# cPD: SCTS for Parkinson's Disorder
# cbvFTD: SCTS for behavioral variant of FrontoTemporal Dementia
# cCP: SCTS for Chronic Pain
# cM: SCTS for Moderate

class MDP():
    """
    MDP game setting
    """
    def __init__(self,algorithm,reward_functions,nTrials,T,nAct_B=20,nAct_C=20,GAMMA=0.95):

        self.nTrials= nTrials # number of experiments to run, large number means longer execution time
        self.T = T            # number of episodes per experiment, large number means longer execution time
        self.algorithm = algorithm
        self.STATE_A,self.STATE_B,self.STATE_C,self.STATE_D,self.STATE_E = 1,2,3,4,5
        self.ACTION_LEFT,self.ACTION_RIGHT,self.ACTION_DUMMY = 0,1,-1
        self.GAMMA = GAMMA
        self.reward_from_B = reward_functions[0]
        self.reward_from_C = reward_functions[1]
        self.nArms = 2
        self.initialState = self.STATE_A
        
        # map actions to states
        self.actionsPerState = {}
        self.actionsPerState[self.STATE_A] = [self.ACTION_LEFT, self.ACTION_RIGHT]
        self.actionsPerState[self.STATE_B] = [i for i in range(nAct_B)]
        self.actionsPerState[self.STATE_C] = [i for i in range(nAct_C)]
        self.actionsPerState[self.STATE_D] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_E] = [self.ACTION_DUMMY]
        self.stateSpace = [self.STATE_A,self.STATE_B,self.STATE_C,self.STATE_D,self.STATE_E]
        fQprime = None
        
        # init Q values
        self.Q1,self.Q2 = {},{}
        
    def getQaddition(self,pw=1,nw=1):
        Qprime = {}
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                Qprime[s][a] = pw * self.Q1[s][a] + nw * self.Q2[s][a]
        return Qprime

    def getQbeta(self):
        Qprime = {}
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                q1 = self.Q1[s][a] if self.Q1[s][a] > 0 else 1
                q2 = self.Q2[s][a] if self.Q2[s][a] > 0 else 1
                Qprime[s][a] = np.random.beta(q1,q2)
        return Qprime

    def getQUCB(self,N,NSA,current_a):
        Qprime = {}
        for s in self.stateSpace:
            Qprime[s] = {}
            if N[s] < len(self.actionsPerState[s]):
                for a in self.actionsPerState[s]:
                    Qprime[s][a] = 0
                Qprime[s][self.actionsPerState[s][N[s]]] = 1
            else:
                for a in self.actionsPerState[s]:
                    if a == current_a:
                        Qprime[s][a] = self.Q1[s][a] * (NSA[s][a]-1) / NSA[s][a] + np.sqrt(2*np.log(N[s])/NSA[s][a])
                    else:
                        Qprime[s][a] = self.Q1[s][a] + np.sqrt(2*np.log(N[s])/NSA[s][a])

        return Qprime
    
    def getQLinUCB(self,N,NSA,current_a):
        Qprime = {}
        for s in self.stateSpace:
            Qprime[s] = {}
            if N[s] < len(self.actionsPerState[s]):
                for a in self.actionsPerState[s]:
                    Qprime[s][a] = 0
                Qprime[s][self.actionsPerState[s][N[s]]] = 1
            else:
                for a in self.actionsPerState[s]:
                    Qprime[s][a] = NSA[s][a] * self.Q1[s][a] + np.sqrt(NSA[s][a])
        return Qprime
    
    def getQCTS(self,Q,NSA):
        gR = 0.5
        gepsilon = 0.05
        gdelta = 0.1
        v2 = (gR**2) * 24 * 1 * math.log(1./gdelta) * (1./gepsilon)
        Qprime = {}
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                Qprime[s][a] = np.random.normal(NSA[s][a] * Q[s][a], v2 * NSA[s][a])
        return Qprime
        
    def getQSCTS(self,NSA,pw=1,nw=1):
        Qprime = {}
        Q1t = self.getQCTS(self.Q1,NSA)
        Q2t = self.getQCTS(self.Q2,NSA)
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                Qprime[s][a] = pw*Q1t[s][a] + nw*Q2t[s][a]
        return Qprime
    
    def getQGreedy(self,N,NSA,current_a):
        Qprime = {}
        for s in self.stateSpace:
            Qprime[s] = {}
            if N[s] < len(self.actionsPerState[s]):
                for a in self.actionsPerState[s]:
                    Qprime[s][a] = 0
                Qprime[s][self.actionsPerState[s][N[s]]] = 1
            else:
                for a in self.actionsPerState[s]:
                    if a == current_a:
                        Qprime[s][a] = self.Q1[s][a] * (NSA[s][a]-1) / NSA[s][a]
                    else:
                        Qprime[s][a] = self.Q1[s][a]
        return Qprime
   
    def getQEXP3(self,mu):
        Qprime = {}
        for s in self.stateSpace:
            Qprime[s] = {}
            for a in self.actionsPerState[s]:
                Qprime[s][a] = (1-mu) * self.Q1[s][a] / np.sum(list(self.Q1[s].values())) + mu / len(self.actionsPerState[s])
        return Qprime

    def resetQprimeFunction(self):
        if self.algorithm in ['DQL','QL','SARSA','MP','SQL','SQL2','PQL','NQL','MP','ESQL','DSQL','ADD','ADHD','AD','CP','bvFTD','PD','M']:
            self.fQprime = self.getQaddition
        elif self.algorithm in ['UCB']:
            self.fQprime = self.getQUCB
        elif self.algorithm in ['LinUCB']:
            self.fQprime = self.getQLinUCB
        elif self.algorithm in ['eGreedy']:
            self.fQprime = self.getQGreedy
        elif self.algorithm in ['EXP3','EXP30','EXP4']:
            self.fQprime = self.getQEXP3
        elif self.algorithm in ['CTS']:
            self.fQprime = self.getQCTS
        elif self.algorithm in ['SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
            self.fQprime = self.getQSCTS
        else: 
            self.fQprime = self.getQbeta
    
    # reset the variables, to be called on each experiment
    def reset(self):
        self.resetQprimeFunction()
        if self.algorithm in ['DQL','QL','SARSA','MP','SQL','SQL2','PQL','NQL','MP','ESQL','DSQL','ADD','ADHD','AD','CP','bvFTD','PD','M','CTS','SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
            defaultQ = 0
        else:
            defaultQ = 1
        for s in self.stateSpace:
            self.Q1[s],self.Q2[s] = {},{}
            for a in self.actionsPerState[s]:
                self.Q1[s][a] = self.Q2[s][a] = defaultQ
      
    # epsilon greedy action
    def random_action(self,s,a,eps=.1):
        p = np.random.random()
        if p < (1 - eps): return a
        else: return np.random.choice(self.actionsPerState[s])
      
    # move from state s using action a
    def move(self,s,a):
        if(s==self.STATE_A):
            if(a == self.ACTION_LEFT): return 0, self.STATE_B
            elif(a == self.ACTION_RIGHT): return 0, self.STATE_C
        if s==self.STATE_B: return self.reward_from_B(), self.STATE_D
        if s==self.STATE_C: return self.reward_from_C(), self.STATE_E
        return 0, s
    
    # returns the action that makes the max Q value, as well as the max Q value
    def maxQA(self,q,s):
        max=float('-inf')
        sa = 0
        if len(q[s]) == 1:
            for k in q[s]: return k,q[s][k]
        for k in q[s]:
            if(q[s][k] > max):
                max = q[s][k]
                sa = k
            elif(q[s][k] == max):
                if(np.random.random() < 0.5): sa = k
        return sa, max
    
    # return true if this is a terminal state
    def isTerminal(self,s):
        return s == self.STATE_E or s == self.STATE_D

    # do the experiment by running T episodes and fill the results in the episodes parameter
    def experiment(self):
        episodes = {} 
        self.reset()
        ALeft = 0 #contains the number of times left action is chosen at A
        
        N = {} # contains the number of visits for each state
        for s in self.stateSpace: N[s] = 0

        NSA = {} # contains the number of visits for each state and action
        for s in self.stateSpace: 
            NSA[s] = {}
            for a in self.actionsPerState[s]:
                NSA[s][a] = 0
        
        t = 0    
        reward,pos_reward,neg_reward,actions = None,None,None,None

        last_a = None
            
        # loop for T episodes
        for i in range(self.T):
            gameover = False
            
            s = self.initialState
            if i == 0: a = self.selectInitialAction(self.initialState,True,last_a,N,NSA)
            else: a = self.selectInitialAction(self.initialState,False,last_a,N,NSA)
            
            #loop until game is over, this will be ONE episode
            while not gameover:
                actions = a
                t += 1  # record learning steps
                
                if self.algorithm in ['DQL','QL','SARSA','MP','SQL','SQL2','PQL','NQL','MP','ESQL','DSQL','ADD','ADHD','AD','CP','bvFTD','PD','M']:
                    a = self.random_action(s, a, 0.05) # apply epsilon greedy selection (including for action chosen at STATE A)
                if self.algorithm == 'eGreedy' and N[s] > len(self.actionsPerState[s]):
                    a = self.random_action(s, a, 0.05) # apply epsilon greedy selection (including for action chosen at STATE A)
                
                N[s] += 1 #update the number of visits for state s

                # if left action is chosen at state A, increment the counter
                if (s == self.STATE_A and a == self.ACTION_LEFT): ALeft += 1

                #move to the next state and get the reward
                r, nxt_s = self.move(s,a)
                reward = r
                if r > 0: pos_reward,neg_reward = r,0
                else: pos_reward,neg_reward = 0,r                
#                 reward.append(r)

                #update the number of visits per state and action
                if not s in NSA: NSA[s] = {}
                NSA[s][a] += 1

                #compute alpha
                alpha = 1 / np.power(NSA[s][a], .8)

                #update the agents and get the best action for the next state
                nxt_a = self.updateAgent(s, a, r, nxt_s, alpha, t, N=N, NSA=NSA)             

                #if next state is terminal then mark as gameover (end of episode)
                gameover = self.isTerminal(nxt_s)
                last_a = a
                s, a = nxt_s, nxt_a

            #update stats for each episode
            if not (i in episodes):
                episodes[i] = {}
                episodes[i]["count"] = 0
                episodes[i]["Q1(A)l"] = episodes[i]["Q1(A)r"] = episodes[i]["Q2(A)l"] = episodes[i]["Q2(A)r"] = 0
            
            episodes[i]["count"],episodes[i]["percent"] = ALeft, ALeft/(i+1)
            episodes[i]["reward"] = reward
            episodes[i]["pos_reward"],episodes[i]["neg_reward"] = pos_reward,neg_reward
            episodes[i]["actions"] = actions
#             episodes[i]["cumreward"] = sum(reward)
            episodes[i]["Q1(A)l"] = (episodes[i]["Q1(A)l"] * i + self.Q1[self.STATE_A][self.ACTION_LEFT])/(i+1)
            episodes[i]["Q2(A)l"] = (episodes[i]["Q2(A)l"] * i + self.Q2[self.STATE_A][self.ACTION_LEFT])/(i+1)
            episodes[i]["Q1(A)r"] = (episodes[i]["Q1(A)r"] * i + self.Q1[self.STATE_A][self.ACTION_RIGHT])/(i+1)
            episodes[i]["Q2(A)r"] = (episodes[i]["Q2(A)r"] * i + self.Q2[self.STATE_A][self.ACTION_RIGHT])/(i+1)
        
        return episodes 
        
    # run the learning
    def run(self):
        #run batch of experiments
        report = {}
        count  = np.ndarray((self.nTrials,self.T))
        percent = np.ndarray((self.nTrials,self.T))
        Q1Al = np.ndarray((self.nTrials,self.T))
        Q2Al = np.ndarray((self.nTrials,self.T))
        Q1Ar = np.ndarray((self.nTrials,self.T))
        Q2Ar = np.ndarray((self.nTrials,self.T))
        reward = np.ndarray((self.nTrials,self.T))
        cumreward = pos_reward = neg_reward = actions = np.ndarray((self.nTrials,self.T))

        for k in range(self.nTrials):
            tmp = self.experiment()
        
            #aggregate every experiment result into the final report
            for i in range(self.T):
                count[k,i] = tmp[i]["count"]
                percent[k,i] = 100*tmp[i]["count"] / (i+1)
                Q1Al[k,i] = tmp[i]["Q1(A)l"]
                Q2Al[k,i] = tmp[i]["Q2(A)l"]
                Q1Ar[k,i] = tmp[i]["Q1(A)r"]
                Q2Ar[k,i] = tmp[i]["Q2(A)r"]
#                 cumreward[k,i] = tmp[i]["cumreward"]
                reward[k,i] = tmp[i]["reward"]
                pos_reward[k,i] = tmp[i]["pos_reward"]
                neg_reward[k,i] = tmp[i]["neg_reward"]
                actions[k,i] = tmp[i]["actions"]
            
        report["count"],report["percent"]  = count,percent
        report["Q1(A)l"],report["Q2(A)l"],report["Q1(A)r"],report["Q2(A)r"] = Q1Al,Q2Al,Q1Ar,Q2Ar
#         report["cumreward"] = cumreward
        report["reward"] = reward
        report["pos_reward"],report["neg_reward"],report["actions"] = pos_reward,neg_reward,actions
        
        return report

    def selectInitialAction(self,startState,veryFirst=False,last_a=None,N=None,NSA=None):
        if veryFirst:
            return np.random.choice(self.actionsPerState[startState])
        else:
            a,_,_ = self.act(startState,last_a,N,NSA)
            return a

    def draw(self,p,actions):
        a = np.random.choice(actions, 1, p)[0]
        return a
    
    def getExpected(self,Q,s): 
        r = []
        for q in Q[s]: r.append(q)
        return np.mean(r)

    def getBias(self):
        if self.algorithm in ['SQL','SQL2','HBTS','MP','ESQL','EHBTS','SCTS']: p1,p2,n1,n2 = 1,1,1,1
        elif self.algorithm in ['PQL','PTS','PCTS']: p1,p2,n1,n2 = 1,1,0,0
        elif self.algorithm in ['NQL','NTS','NCTS']: p1,p2,n1,n2 = 0,0,1,1 
        elif self.algorithm in ['ADD','bADD','cADD']: p1,p2,n1,n2 = np.random.normal(1,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        elif self.algorithm in ['ADHD','bADHD','cADHD']: p1,p2,n1,n2 = np.random.normal(0.2,0.1),np.random.normal(1,0.1),np.random.normal(0.2,0.1),np.random.normal(1,0.1)
        elif self.algorithm in ['AD','bAD','cAD']: p1,p2,n1,n2 = np.random.normal(0.1,0.1),np.random.normal(1,0.1),np.random.normal(0.1,0.1),np.random.normal(1,0.1)
        elif self.algorithm in ['CP','bCP','cCP']: p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(1,0.1)
        elif self.algorithm in ['bvFTD','bbvFTD','cbvFTD']: p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(100,10),np.random.normal(0.5,0.1),np.random.normal(1,0.1)  
        elif self.algorithm in ['PD','bPD','cPD']: p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(100,10)
        elif self.algorithm in ['M','bM','cM']: p1,p2,n1,n2 = np.random.normal(0.5,0.1),np.random.normal(1,0.1),np.random.normal(0.5,0.1),np.random.normal(1,0.1)
        else: p1,p2,n1,n2 = None,None,None,None
        return p1,p2,n1,n2
        
    def act(self,s,last_a,N,NSA):
        
        maxq = None
        isQ1forDQL = False
        
        p1,p2,n1,n2 = self.getBias()

        if self.algorithm in ['SARSA','QL']:
            nxt_a, maxq = self.maxQA(self.Q1, s)    

        if self.algorithm in ['DQL','EDQL']:
            p = np.random.random()
            if (p < .5): 
                nxt_a, maxq = self.maxQA(self.Q1, s)
                isQ1forDQL = True
            else: 
                nxt_a, maxq = self.maxQA(self.Q2, s)
                isQ1forDQL = False

        if self.algorithm == 'MP':
            Qprime = self.fQprime()
            nxt_a, maxq = self.maxQA(Qprime, s)
                                   
        if self.algorithm in ['SQL','ESQL','PQL','NQL','AD','ADD','ADHD','CP','bvFTD','PD','M']:
            Qprime = self.fQprime()
            nxt_a, maxq = self.maxQA(Qprime, s)
            
        if self.algorithm in ['SQL2']:
            Qprime = self.fQprime(pw=p2,nw=n2)
            nxt_a, maxq = self.maxQA(Qprime, s)
            
        if self.algorithm in ['EXP3','EXP4']:
            mu = 0.05
            Qprime = self.fQprime(mu)
            p = list(Qprime[s].values())
            nxt_a = self.draw(p,self.actionsPerState[s])

        if self.algorithm in ['EXP30']:
            mu = 0
            Qprime = self.fQprime(mu)
            p = list(Qprime[s].values())
            nxt_a = self.draw(p,self.actionsPerState[s])

        if self.algorithm in ['eGreedy','UCB','LinUCB']:
            Qprime = self.fQprime(N,NSA,last_a)
            nxt_a, maxq = self.maxQA(Qprime, s)
            
        if self.algorithm in ['SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
            Qprime = self.fQprime(NSA)
            nxt_a, maxq = self.maxQA(Qprime, s)

        if self.algorithm in ['CTS']:
            Qprime = self.fQprime(self.Q1,NSA)
            nxt_a, maxq = self.maxQA(Qprime, s)
            
        if self.algorithm in ['TS','ETS','HBTS','PTS','NTS','EHBTS','bADD','bADHD','bAD','bCP','bbvFTD','bPD','bM']:
            Qprime = self.fQprime()
            nxt_a, maxq = self.maxQA(Qprime, s)
                
        return nxt_a, maxq, isQ1forDQL
    
    def updateAgent(self, s, a, r, nxt_s, alpha, t, pr=None, nr=None, N=None, NSA=None):
        
        p1,p2,n1,n2 = self.getBias()
        
        nxt_a, maxq, isQ1forDQL = self.act(nxt_s,a,N,NSA)

        if self.algorithm == 'SARSA':
            # self.Q1[s][a] = self.Q1[s][a] + alpha * (r + self.GAMMA * self.getExpected(Q1s) - self.Q1[s][a])
            self.Q1[s][a] = self.Q1[s][a] + alpha * (r + self.GAMMA * self.Q1[nxt_s][nxt_a] - self.Q1[s][a])
        
        if self.algorithm == 'QL':
            self.Q1[s][a] = self.Q1[s][a] + alpha * (r + self.GAMMA * maxq - self.Q1[s][a])
    
        if self.algorithm == 'DQL':
            p = np.random.random()
            if isQ1forDQL: self.Q1[s][a] = self.Q1[s][a] + alpha * (r + self.GAMMA * self.Q2[nxt_s][nxt_a] - self.Q1[s][a])
            else: self.Q2[s][a] = self.Q2[s][a] + alpha * (r + self.GAMMA * self.Q1[nxt_s][nxt_a] - self.Q2[s][a])

        if self.algorithm == 'MP':
            if pr or nr:
                self.Q1[s][a] = p1*self.Q1[s][a] + alpha * (p2*pr + self.GAMMA * self.Q1[nxt_s][nxt_a] - self.Q1[s][a])
                self.Q2[s][a] = n1*self.Q2[s][a] + alpha * (n2*nr + self.GAMMA * self.Q2[nxt_s][nxt_a] - self.Q2[s][a])
            else:
                if (r >= 0): self.Q1[s][a] = p1*self.Q1[s][a] + alpha * (p2*r + self.GAMMA * self.Q1[nxt_s][nxt_a] - self.Q1[s][a])
                if (r <= 0): self.Q2[s][a] = n1*self.Q2[s][a] + alpha * (n2*r + self.GAMMA * self.Q2[nxt_s][nxt_a] - self.Q2[s][a])
                       
        if self.algorithm == 'EDQL':
            rho = 1
            p = np.random.random()
            if (p < .5):
                try: r = r*math.exp(r/rho)
                except: r = r
                self.Q1[s][a] = self.Q1[s][a] + alpha * (r + self.GAMMA * self.Q2[nxt_s][nxt_a] - self.Q1[s][a])
            else:
                try: r = r*math.exp(-r/rho)
                except: r = r
                self.Q2[s][a] = self.Q2[s][a] + alpha * (r + self.GAMMA * self.Q1[nxt_s][nxt_a] - self.Q2[s][a])
    
        if self.algorithm == 'ESQL':
            rho = 1    
            nxt_a1, maxq1 = self.maxQA(self.Q1, nxt_s)
            nxt_a2, maxq2 = self.maxQA(self.Q2, nxt_s)
            if pr or nr:
                try: pr = pr*math.exp(pr/rho)
                except: pr = pr
                self.Q1[s][a] = p1*self.Q1[s][a] + alpha * (p2*pr + self.GAMMA * self.Q1[nxt_s][nxt_a1] - self.Q1[s][a])
                try: nr = nr*math.exp(-nr/rho)
                except: nr = nr
                self.Q2[s][a] = n1*self.Q2[s][a] + alpha * (n2*nr + self.GAMMA * self.Q2[nxt_s][nxt_a2] - self.Q2[s][a])
            else:
                if (r >= 0):
                    try: r = r*math.exp(r/rho)
                    except: r = r
                    self.Q1[s][a] = p1*self.Q1[s][a] + alpha * (p2*r + self.GAMMA * self.Q1[nxt_s][nxt_a1] - self.Q1[s][a])
                if (r <= 0):
                    try: r = r*math.exp(-r/rho)
                    except: r = r    
                    self.Q2[s][a] = n1*self.Q2[s][a] + alpha * (n2*r + self.GAMMA * self.Q2[nxt_s][nxt_a2] - self.Q2[s][a])
        
                    
        if self.algorithm in ['SQL','PQL','NQL','AD','ADD','ADHD','CP','bvFTD','PD','M']:
            nxt_a1, maxq1 = self.maxQA(self.Q1, nxt_s)
            nxt_a2, maxq2 = self.maxQA(self.Q2, nxt_s)
            if pr or nr:
                self.Q1[s][a] = p1*self.Q1[s][a] + alpha * (p2*pr + self.GAMMA * self.Q1[nxt_s][nxt_a1] - self.Q1[s][a])
                self.Q2[s][a] = n1*self.Q2[s][a] + alpha * (n2*nr + self.GAMMA * self.Q2[nxt_s][nxt_a2] - self.Q2[s][a])
            else:
                if (r >= 0): self.Q1[s][a] = p1*self.Q1[s][a] + alpha * (p2*r + self.GAMMA * self.Q1[nxt_s][nxt_a1] - self.Q1[s][a])
                if (r <= 0): self.Q2[s][a] = n1*self.Q2[s][a] + alpha * (n2*r + self.GAMMA * self.Q2[nxt_s][nxt_a2] - self.Q2[s][a])

        if self.algorithm in ['SQL2']:
            nxt_a1, maxq1 = self.maxQA(self.Q1, nxt_s)
            nxt_a2, maxq2 = self.maxQA(self.Q2, nxt_s)
            if pr or nr:
                self.Q1[s][a] = p1*self.Q1[s][a] + alpha * (pr + self.GAMMA * self.Q1[nxt_s][nxt_a1] - self.Q1[s][a])
                self.Q2[s][a] = n1*self.Q2[s][a] + alpha * (nr + self.GAMMA * self.Q2[nxt_s][nxt_a2] - self.Q2[s][a])
            else:
                if (r >= 0): self.Q1[s][a] = p1*self.Q1[s][a] + alpha * (r + self.GAMMA * self.Q1[nxt_s][nxt_a1] - self.Q1[s][a])
                if (r <= 0): self.Q2[s][a] = n1*self.Q2[s][a] + alpha * (r + self.GAMMA * self.Q2[nxt_s][nxt_a2] - self.Q2[s][a])
    
        if self.algorithm in ['EHBTS']:
            rho = 1
            if pr or nr:
                try: pr = pr*math.exp(pr/rho)
                except: pr = pr
                self.Q1[s][a] = p1*self.Q1[s][a] + p2*pr    
                try: nr = nr*math.exp(-nr/rho)
                except: nr = nr
                self.Q2[s][a] = n1*self.Q2[s][a] - n2*nr
            else:
                if (r >= 0):
                    try: r = r*math.exp(r/rho)
                    except: r = r
                    self.Q1[s][a] = p1*self.Q1[s][a] + p2*r 
                if (r <= 0):
                    try: r = r*math.exp(-r/rho)
                    except: r = r
                    self.Q2[s][a] = n1*self.Q2[s][a] - n2*r
 
        if self.algorithm in ['TS']:
            if pr or nr: r = pr + nr    
            if (r >= 0): self.Q1[s][a] = self.Q1[s][a] + 1
            if (r <= 0): self.Q2[s][a] = self.Q2[s][a] + 1

        if self.algorithm in ['ETS']:
            if pr or nr:
                try: pr = pr*math.exp(pr/rho)
                except: pr = pr
                try: nr = nr*math.exp(-nr/rho)
                except: nr = nr
                r = pr + nr    
            if (r >= 0): self.Q1[s][a] = self.Q1[s][a] + 1
            if (r <= 0): self.Q2[s][a] = self.Q2[s][a] + 1
        
        if self.algorithm in ['EXP3']:
            mu = 0.05
            Qprime = self.fQprime(mu)
            est_r = r / Qprime[s][a]
            self.Q1[s][a] = self.Q1[s][a] * np.exp(mu*est_r/len(self.actionsPerState[s]))

        if self.algorithm in ['EXP30']:
            mu = 0
            Qprime = self.fQprime(mu)
            est_r = r / Qprime[s][a]
            self.Q1[s][a] = self.Q1[s][a] * np.exp(mu*est_r/len(self.actionsPerState[s]))
                
        if self.algorithm in ['eGreedy','UCB']:
            self.Q1[s][a] = self.Q1[s][a] + (r - self.Q1[s][a]) / N[s]

        if self.algorithm in ['HBTS','PTS','NTS','EHBTS','bADD','bADHD','bAD','bCP','bbvFTD','bPD','bM']:
            if pr or nr:
                self.Q1[s][a] = p1*self.Q1[s][a] + p2*pr
                self.Q2[s][a] = n1*self.Q2[s][a] - n2*nr 
            else:
                if (r >= 0): self.Q1[s][a] = p1*self.Q1[s][a] + p2*r 
                if (r <= 0): self.Q2[s][a] = n1*self.Q2[s][a] - n2*r 
                
        if self.algorithm in ['EXP4']:
            mu = 0.05
            Qprime = self.fQprime(mu)
            est_r = r / Qprime[s][a]
            for st in self.stateSpace:
                for at in self.actionsPerState[st]:
                    self.Q1[st][at] = self.Q1[st][at] * np.exp(mu*est_r/len(self.actionsPerState[st]))

        if self.algorithm in ['LinUCB','CTS']:
            self.Q1[s][a] = self.Q1[s][a] + r

        if self.algorithm in ['SCTS','PCTS','NCTS','cADD','cADHD','cAD','cCP','cbvFTD','cPD','cM']:
            if pr or nr:
                self.Q1[s][a] = p1*self.Q1[s][a] + p2*pr
                self.Q2[s][a] = n1*self.Q2[s][a] + n2*nr 
            else:
                if (r >= 0): self.Q1[s][a] = p1*self.Q1[s][a] + p2*r 
                if (r <= 0): self.Q2[s][a] = n1*self.Q2[s][a] + n2*r       
          
        return nxt_a


class MAB(MDP):
    """
    MAB game setting
    """
    def __init__(self,algorithm,reward_functions,nTrials,T):
        MDP.__init__(self,algorithm,reward_functions,nTrials,T)
                
        # map actions to states
        self.actionsPerState = {}
        self.actionsPerState[self.STATE_A] = [self.ACTION_LEFT, self.ACTION_RIGHT]
        self.actionsPerState[self.STATE_B] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_C] = [self.ACTION_DUMMY]
        self.stateSpace = [self.STATE_A,self.STATE_B,self.STATE_C]

    # move from state s using action a
    def move(self,s,a):
        if(s==self.STATE_A):
            if(a == self.ACTION_LEFT): return self.reward_from_B(), self.STATE_B
            elif(a == self.ACTION_RIGHT): return self.reward_from_C(), self.STATE_C
        return 0, s

    # return true if this is a terminal state
    def isTerminal(self,s):
        return s == self.STATE_B or s == self.STATE_C


class IGT(MDP):
    """
    IGT game setting
    """
    def __init__(self,algorithm,reward_functions,nTrials,T):
        MDP.__init__(self,algorithm,reward_functions,nTrials,T)
        
        self.reward_from_A = reward_functions[0]
        self.reward_from_B = reward_functions[1]
        self.reward_from_C = reward_functions[2]
        self.reward_from_D = reward_functions[3]
    
        # In IGT, the initial state is self.STATE_E
        self.ACTION_A,self.ACTION_B,self.ACTION_C,self.ACTION_D = 0,1,2,3
        # map actions to states
        self.actionsPerState = {}
        self.actionsPerState[self.STATE_E] = [self.ACTION_A,self.ACTION_B,self.ACTION_C,self.ACTION_D]
        self.actionsPerState[self.STATE_A] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_B] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_C] = [self.ACTION_DUMMY]
        self.actionsPerState[self.STATE_D] = [self.ACTION_DUMMY]
        
        self.nArms = 4
        self.initialState = self.STATE_E    
  
    def move(self,s,a,N): 
        if(s==self.STATE_E):
            if(a == self.ACTION_A):  return self.reward_from_A(N[self.STATE_A]), self.STATE_A
            elif(a == self.ACTION_B): return self.reward_from_B(N[self.STATE_B]), self.STATE_B
            elif(a == self.ACTION_C): return self.reward_from_C(N[self.STATE_C]), self.STATE_C
            elif(a == self.ACTION_D): return self.reward_from_D(N[self.STATE_D]), self.STATE_D
        return [0,0],s
            
    def isTerminal(self,s):
        return s == self.STATE_A or s == self.STATE_B or s == self.STATE_C or s == self.STATE_D

    def experiment(self):   
        episodes = {}
        t = 0
        self.reset()
        ILeft = 0 #contains the number of times left action is chosen at initial state I
        N={}    # contains the number of visits for each state
        for s in self.stateSpace: N[s] = 0
        NSA = {}        # contains the number of visits for each state and action
        for s in self.stateSpace: 
            NSA[s] = {}
            for a in self.actionsPerState[s]:
                NSA[s][a] = 0
        reward,pos_reward,neg_reward,actions = None,None,None,None
        
        last_a = None
        
        # loop for T episodes
        for i in range(self.T):

            s = self.initialState
            
            if i == 0: a = self.selectInitialAction(self.initialState,True,last_a,N,NSA)
            else: a = self.selectInitialAction(self.initialState,False,last_a,N,NSA)

            gameover = False

            #loop until game is over, this will be ONE episode
            while not gameover:
                actions = a
                # record learning steps
                t += 1
                
                if self.algorithm in ['DQL','QL','SARSA','MP','SQL','SQL2','PQL','NQL','MP','ESQL','DSQL','ADD','ADHD','AD','CP','bvFTD','PD','M']:
                    a = self.random_action(s, a, 0.05) # apply epsilon greedy selection (including for action chosen at STATE A)
                if self.algorithm == 'eGreedy' and N[s] > len(self.actionsPerState[s]):
                    a = self.random_action(s, a, 0.05) # apply epsilon greedy selection (including for action chosen at STATE A)

                #update the number of visits for state s
                N[s] += 1

                # if left action is chosen at state A, increment the counter
                if (s == self.STATE_E and a == self.ACTION_A) or (s == self.STATE_E and a == self.ACTION_B):
                    ILeft += 1

                #move to the next state and get the reward
                [pr,nr], nxt_s = self.move(s,a,N)
                r = pr+nr
                reward,pos_reward,neg_reward = r,pr,nr

                #update the number of visits per state and action
                if not s in NSA: NSA[s] = {}
                NSA[s][a] += 1

                #compute alpha
                alpha = 1 / np.power(NSA[s][a], .8)

                #update the Q values and get the best action for the next state
                nxt_a = self.updateAgent(s, a, r, nxt_s, alpha, t, pr, nr, N, NSA)

                #if next state is terminal then mark as gameover (end of episode)
                gameover = self.isTerminal(nxt_s)

                if gameover: N[nxt_s] += 1
                last_a = a
                s = nxt_s
                a = nxt_a

            #update stats for each episode
            if not (i in episodes):
                episodes[i] = {}
                episodes[i]["count"] = 0
                episodes[i]["Q1(I)a"] = episodes[i]["Q2(I)a"] = episodes[i]["Q1(I)b"] = episodes[i]["Q2(I)b"] = 0
                episodes[i]["Q1(I)c"] = episodes[i]["Q2(I)c"] = episodes[i]["Q1(I)d"] = episodes[i]["Q2(I)d"] = 0
            episodes[i]["count"],episodes[i]["percent"] = ILeft, ILeft / (i+1)
            episodes[i]["reward"],episodes[i]["pos_reward"],episodes[i]["neg_reward"] = reward,pos_reward,neg_reward
            episodes[i]["actions"] = actions
#             episodes[i]["cumreward"] = sum(reward)
            episodes[i]["Q1(I)a"] = ((episodes[i]["Q1(I)a"] * i) + self.Q1[self.STATE_E][self.ACTION_A])/(i+1)
            episodes[i]["Q2(I)a"] = ((episodes[i]["Q2(I)a"] * i) + self.Q2[self.STATE_E][self.ACTION_A])/(i+1)
            episodes[i]["Q1(I)b"] = ((episodes[i]["Q1(I)b"] * i) + self.Q1[self.STATE_E][self.ACTION_B])/(i+1)
            episodes[i]["Q2(I)b"] = ((episodes[i]["Q2(I)b"] * i) + self.Q2[self.STATE_E][self.ACTION_B])/(i+1)
            episodes[i]["Q1(I)c"] = ((episodes[i]["Q1(I)c"] * i) + self.Q1[self.STATE_E][self.ACTION_C])/(i+1)
            episodes[i]["Q2(I)c"] = ((episodes[i]["Q2(I)c"] * i) + self.Q2[self.STATE_E][self.ACTION_C])/(i+1)
            episodes[i]["Q1(I)d"] = ((episodes[i]["Q1(I)d"] * i) + self.Q1[self.STATE_E][self.ACTION_D])/(i+1)
            episodes[i]["Q2(I)d"] = ((episodes[i]["Q2(I)d"] * i) + self.Q2[self.STATE_E][self.ACTION_D])/(i+1)
        
        return episodes 
   
    def run(self):
        report = {}
        count = percent = np.ndarray((self.nTrials,self.T))
        Q1Ia = np.ndarray((self.nTrials,self.T))
        Q2Ia = np.ndarray((self.nTrials,self.T))
        Q1Ib = np.ndarray((self.nTrials,self.T))
        Q2Ib = np.ndarray((self.nTrials,self.T))
        Q1Ic = np.ndarray((self.nTrials,self.T))
        Q2Ic = np.ndarray((self.nTrials,self.T))
        Q1Id = np.ndarray((self.nTrials,self.T))
        Q2Id = np.ndarray((self.nTrials,self.T))
        cumreward = reward = pos_reward = neg_reward = actions = np.ndarray((self.nTrials,self.T))
    
        #run batch of experiments
        for k in range(self.nTrials):
            tmp = self.experiment()
            #aggregate every experiment result into the final report
            for i in range(self.T):
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
#                 cumreward[k,i] = tmp[i]["cumreward"]
                reward[k,i] = tmp[i]["reward"]
                pos_reward[k,i] = tmp[i]["pos_reward"]
                neg_reward[k,i] = tmp[i]["neg_reward"]
                actions[k,i] = tmp[i]["actions"]

        report["count"],report["percent"] = count,percent
        report["Q1(I)a"],report["Q2(I)a"],report["Q1(I)b"],report["Q2(I)b"] = Q1Ia,Q2Ia,Q1Ib,Q2Ib
        report["Q1(I)c"],report["Q2(I)c"],report["Q1(I)d"],report["Q2(I)d"] = Q1Ic,Q2Ic,Q1Id,Q2Id
        report["cumreward"],report["reward"],report["pos_reward"],report["neg_reward"] = cumreward,reward,pos_reward,neg_reward
        report["actions"] = actions
        return report
    
