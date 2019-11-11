# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from pacman import Rmax, Rmin
from learningAgents import ReinforcementAgent
from featureExtractors import *
from numpy.random import beta

import random,util,math

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from random import randint
from math import log, sqrt
gp = GaussianProcessRegressor()

# BL ADDED: to specify reward range for EXP3 and GPUCB
# TIME_PENALTY = -1 # Number of points lost each round
# FOOD_BONUS = 10 # Bonus for eating a food
# WIN_BONUS = 500 # Bonus for winning the game
# GHOST_BONUS = 200   # Bonus of eating a ghost
# LOSE_PENALTY = -500  # Penalty for losing the game/touching a ghost
# Rmax = TIME_PENALTY + GHOST_BONUS + WIN_BONUS + FOOD_BONUS
# Rmin = TIME_PENALTY + LOSE_PENALTY

# import sys
# import threading

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
    """
    def __init__(self, **args):
        "Initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_vals = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_vals[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:  # Empty, i.e. no legal actions
            return 0.0

        max_val = -float('inf')
        for a in legalActions:
            max_val = max(max_val, self.getQValue(state, a))
        return max_val

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          return None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:  # Empty, i.e. no legal actions
            return None

        max_val = -float('inf')
        best_acts = []
        for a in legalActions:
            this_q = self.getQValue(state, a)
            if(this_q > max_val):
                best_acts = [a]
                max_val = this_q
            elif(this_q == max_val):
                best_acts.append(a)
        return random.choice(best_acts)


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, 
          choose None as the action.
        """
        legalActions = self.getLegalActions(state)
        action = None
        
        if not legalActions:
            return None
        r = util.flipCoin(self.epsilon)
        if r:
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward, positiveReward, negativeReward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          Q-Value update here
        """
        cum_R = reward + self.discount * self.computeValueFromQValues(nextState)
        self.q_vals[(state, action)] += self.alpha * (cum_R - self.getQValue(state, action))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2,p1=1.0,p2=1.0,n1=1.0,n2=1.0,lr=1e-4,pw=1.0,nw=1.0,ucbbeta=100.,exp3gamma=0.5,numTraining=0,ucbalpha=1.,ucbwindow=1000,byround=False, **args):  # BL: Add p1, p2, n1, n2 here to enable TS
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining

        # BL: Add p1, p2, n1, n2 here to enable TS
        # BL: Add lr, pw, nw to constrain it
        args['p1'] = p1 
        args['p2'] = p2
        args['n1'] = n1
        args['n2'] = n2
        args['lr'] = lr
        args['pw'] = pw
        args['nw'] = nw
        args['ucbbeta'] = ucbbeta
        args['exp3gamma'] = exp3gamma
        args['byround'] = byround
        args['ucbalpha'] = ucbalpha
        args['ucbwindow'] = ucbwindow

        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class QL(PacmanQAgent):
    """
       Q Learning Agent
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        feats = self.featExtractor.getFeatures(state, action)
        q = 0
        for i in feats:
            q += feats[i] * self.weights[i]
        return q

    def update(self, state, action, nextState, reward, positiveReward, negativeReward):
        feats = self.featExtractor.getFeatures(state, action)
        cum_R = reward + self.discount * self.computeValueFromQValues(nextState)
        difference = cum_R - self.getQValue(state, action)
        for i in feats:
            self.weights[i] += self.alpha * difference * feats[i]

    def final(self, state):
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass


#########################################################
# BL ADDED: CLASS WRITTEN FOR Double Q-LEARN AGENT      #
#########################################################

class DQL(PacmanQAgent):
    """
       Double Q Learning Agent
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights_1 = util.Counter()
        self.weights_2 = util.Counter()
        self.qnow = 1

    def getWeights(self):
        return [self.weights_1,self.weights_2]

    def getQValue(self, state, action):
        feats = self.featExtractor.getFeatures(state, action)
        q = 0
        for i in feats:
            if self.qnow == 1:
                q += feats[i] * self.weights_1[i]
            else:
                q += feats[i] * self.weights_2[i]
        return q

    def getAction(self, state):
        r = util.flipCoin(0.5)
        if r:
            self.qnow = 1
        else:
            self.qnow = 2
        action = PacmanQAgent.getAction(self,state)
        self.doAction(state,action)
        return action
    
    def update(self, state, action, nextState, reward, positiveReward, negativeReward):
        feats = self.featExtractor.getFeatures(state, action)
        cum_R = reward + self.discount * self.computeValueFromQValues(nextState)
        difference = cum_R - self.getQValue(state, action)
        for i in feats:
            if self.qnow == 1:
                self.weights_2[i] += self.alpha * difference * feats[i]
            else:
                self.weights_1[i] += self.alpha * difference * feats[i]

    def final(self, state):
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass


#########################################################
# BL ADDED: CLASS WRITTEN FOR SPLIT Q-LEARN AGENT       #
#########################################################
class SQL(PacmanQAgent):
    """
       Split Q-Learning Agent
    """
    def __init__(self,extractor='IdentityExtractor',**args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.weights_pos = util.Counter()
        self.weights_neg = util.Counter()

    def getWeights(self):
        return [self.weights, self.weights_pos, self.weights_neg]
    
    def getQValue(self, state, action):
        feats = self.featExtractor.getFeatures(state, action)
        q = 0
        for i in feats:
            q += feats[i] * self.weights[i]
        return q
    
    def getPosQValue(self, state, action):
        feats = self.featExtractor.getFeatures(state, action)
        q_pos = 0
        for i in feats:
            q_pos += feats[i] * self.weights_pos[i]
        return q_pos

    def getNegQValue(self, state, action):   
        feats = self.featExtractor.getFeatures(state, action)
        q_neg = 0
        for i in feats:
            q_neg += feats[i] * self.weights_neg[i]
        return q_neg

    def getSplitQValue(self, state, action):
        return self.pw * self.getPosQValue(state, action) + self.nw * self.getNegQValue(state, action)

    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:  # Empty, i.e. no legal actions
            return None
        max_val = -float('inf')
        best_acts = []
        for a in legalActions:
            this_q = self.getQValue(state, a)
            if(this_q > max_val):
                best_acts = [a]
                max_val = this_q
            elif(this_q == max_val):
                best_acts.append(a)
        return random.choice(best_acts)
    
    def computeValueFromPosQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:  # Empty, i.e. no legal actions
            return 0.0

        max_val = -float('inf')
        for a in legalActions:
            max_val = max(max_val, self.getPosQValue(state, a))
        return max_val

    def computeValueFromNegQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:  # Empty, i.e. no legal actions
            return 0.0

        max_val = -float('inf')
        for a in legalActions:
            max_val = max(max_val, self.getNegQValue(state, a))
        return max_val

    def computeActionFromSplitQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:  # Empty, i.e. no legal actions
            return None

        max_val = -float('inf')
        best_acts = []
        for a in legalActions:
            this_q = self.getSplitQValue(state, a)
            if(this_q > max_val):
                best_acts = [a]
                max_val = this_q
            elif(this_q == max_val):
                best_acts.append(a)
        return random.choice(best_acts)

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        action = None
        
        if not legalActions:
            return None
        r = util.flipCoin(self.epsilon)
        if r:
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromSplitQValues(state)
        self.doAction(state,action)
        return action

    def update(self, state, action, nextState, reward, positiveReward, negativeReward):
        feats = self.featExtractor.getFeatures(state, action)
        
        difference_qp = self.p2 * positiveReward + self.discount * self.computeValueFromPosQValues(nextState) - self.getPosQValue(state, action)
        difference_qn = self.n2 * negativeReward + self.discount * self.computeValueFromNegQValues(nextState) - self.getNegQValue(state, action)
        
        for i in feats:
            self.weights_pos[i] = self.p1 * self.weights_pos[i] + self.alpha * difference_qp * feats[i]
            self.weights_neg[i] = self.n1 * self.weights_neg[i] + self.alpha * difference_qn * feats[i]

    def final(self, state):
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass
                
#########################################################
# BL ADDED: CLASS WRITTEN FOR POSITIVE Q-LEARN AGENT    #
#########################################################

class PQL(SQL):
    """
       Positive Q-Learning Agent
    """
    def __init__(self,extractor='IdentityExtractor',**args):
        self.featExtractor = util.lookup(extractor, globals())()
        SQL.__init__(self, **args)
        self.weights_pos = util.Counter()
        self.weights_neg = util.Counter()
        self.n2 = 0.0

#########################################################
# BL ADDED: CLASS WRITTEN FOR NEGATIVE Q-LEARN AGENT    #
#########################################################

class NQL(SQL):
    """
       Negative Q-Learning Agent
    """
    def __init__(self,extractor='IdentityExtractor',**args):
        self.featExtractor = util.lookup(extractor, globals())()
        SQL.__init__(self, **args)
        self.weights_pos = util.Counter()
        self.weights_neg = util.Counter()
        self.p2 = 0.0

#############################################################
# BL ADDED: CLASS WRITTEN FOR GPUCB SPLIT Q-LEARN AGENT     #
#############################################################

class GPUCBSQL(SQL):
    """
       GPUCB-based Split Q-Learning Agent
    """
    def __init__(self,extractor='IdentityExtractor',**args):
        self.featExtractor = util.lookup(extractor, globals())()
        SQL.__init__(self, **args)
        self.weights_pos = util.Counter()
        self.weights_neg = util.Counter()
        self.iterations = 1

        p1range=np.arange(0,2.1,0.5)
        p2range=np.arange(0,2.1,0.5)
        pwrange=np.array([1])
        n1range=np.arange(0,2.1,0.5)
        n2range=np.arange(0,2.1,0.5)
        nwrange=np.array([1])
        
        self.meshgrid = np.array(np.meshgrid(p1range,p2range,pwrange,n1range,n2range,nwrange))

        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])
        self.X = []
        self.T = []

        self.p1 = None
        self.p2 = None
        self.pw = None
        self.n1 = None
        self.n2 = None
        self.nw = None

    def argmax_ucb(self):
        return np.argmax(self.mu + self.sigma * np.sqrt(self.ucbbeta))

    def computeActionFromSplitQValues(self, state):
             
        grid_idx = self.argmax_ucb()
        x = self.X_grid[grid_idx]
        self.X.append(x)

        self.p1 = x[0]
        self.p2 = x[1]
        self.pw = x[2]
        self.n1 = x[3]
        self.n2 = x[4]
        self.nw = x[5]

        print 'p1: ', x[0]
        print 'p2: ', x[1]
        print 'pw: ', x[2]
        print 'n1: ', x[3]
        print 'n2: ', x[4]
        print 'nw: ', x[5]
        print x

        legalActions = self.getLegalActions(state)
        if not legalActions:  # Empty, i.e. no legal actions
            return None

        max_val = -float('inf')
        best_acts = []
        for a in legalActions:
            this_q = self.getSplitQValue(state, a)
            if(this_q > max_val):
                best_acts = [a]
                max_val = this_q
            elif(this_q == max_val):
                best_acts.append(a)
        return random.choice(best_acts)

    def update(self, state, action, nextState, reward, positiveReward, negativeReward):
        
        feats = self.featExtractor.getFeatures(state, action)
        
        cum_nR = self.n2 * negativeReward + self.discount * self.computeValueFromNegQValues(nextState)
        cum_pR = self.p2 * positiveReward + self.discount * self.computeValueFromPosQValues(nextState)
        difference_qn = cum_nR - self.getNegQValue(state, action)
        difference_qp = cum_pR - self.getPosQValue(state, action)
        for i in feats:
            self.weights_neg[i] = self.n1 * self.weights_neg[i] + self.alpha * difference_qn * feats[i]
            self.weights_pos[i] = self.p1 * self.weights_pos[i] + self.alpha * difference_qp * feats[i]

        self.iterations += 1;

        ucbreward = (reward - Rmin) / (Rmax - Rmin)

        self.T.append(ucbreward)
        try:
            gp.fit(self.X, self.T)
        except ValueError:
            self.T = self.T[:-1]
            gp.fit(self.X, self.T)
        
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)

##############################################################
# BL ADDED: CLASS WRITTEN FOR full GPUCB SPLIT Q-LEARN AGENT #
##############################################################

class fGPUCBSQL(GPUCBSQL):
    """
       Full GPUCB-based Split Q-Learning Agent
    """
    def __init__(self,extractor='IdentityExtractor',**args):
        self.featExtractor = util.lookup(extractor, globals())()
        GPUCBSQL.__init__(self, **args)

        p1range=np.arange(0,2.1,0.5)
        p2range=np.arange(0,2.1,0.5)
        pwrange=np.arange(0,2.1,0.5)
        n1range=np.arange(0,2.1,0.5)
        n2range=np.arange(0,2.1,0.5)
        nwrange=np.arange(0,2.1,0.5)
        
        self.meshgrid = np.array(np.meshgrid(p1range,p2range,pwrange,n1range,n2range,nwrange))

        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])


##############################################################
# BL ADDED: CLASS WRITTEN FOR Meta SPLIT Q-LEARN AGENT       #
##############################################################

class MSQL(SQL):
    """
       Meta Split Q-Learning Agent
    """
    def __init__(self,extractor='IdentityExtractor',**args):
        
        self.featExtractor = util.lookup(extractor, globals())()
        SQL.__init__(self, **args)
        self.iterations = 1
        self.round = 0

        # for policy selection
        self.n_policies = 3
        self.policyrecord = []
 
    def selectPolicy(self):
        raise NotImplementedError("selectPolicy not implemented for Meta SQL")

    def getWeights(self):
        return [self.weights_pos, self.weights_neg, self.weights]

    def getBestQValue(self, state, action):
        
        if not self.byround:
            self.selected_policy = self.selectPolicy()
        print 'selected arm: ', self.selected_policy
        self.policyrecord.append(self.selected_policy)

        if self.selected_policy == 0:
            return self.getSplitQValue(state, action)
        if self.selected_policy == 1:
            return self.getPosQValue(state, action)
        if self.selected_policy == 2:
            return self.getNegQValue(state, action)

    def computeActionFromBestQValues(self, state):
        
        legalActions = self.getLegalActions(state)
        if not legalActions:  # Empty, i.e. no legal actions
            return None

        max_val = -float('inf')
        best_acts = []
        for a in legalActions:
            this_q = self.getBestQValue(state, a)
            if(this_q > max_val):
                best_acts = [a]
                max_val = this_q
            elif(this_q == max_val):
                best_acts.append(a)
        return random.choice(best_acts)

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        action = None
        
        if not legalActions:
            return None
        r = util.flipCoin(self.epsilon)
        if r:
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromBestQValues(state)
        self.doAction(state,action)
        return action
    
    def final(self, state):

        PacmanQAgent.final(self, state)
        self.round = self.round + 1
        
        if self.byround:
            self.selected_policy = self.selectPolicy()

        if self.episodesSoFar == self.numTraining:            
            pass

##############################################################
# BL ADDED: CLASS WRITTEN FOR EXP3-based SPLIT Q-LEARN AGENT #
##############################################################

class EXP3SQL(MSQL):
    """
       EXP3-based Split Q-Learning Agent
    """
    def __init__(self,extractor='IdentityExtractor',**args):
        
        self.featExtractor = util.lookup(extractor, globals())()
        MSQL.__init__(self, **args)

        # for EXP3
        self.exp3weights = [1.0 for i in range(self.n_policies)]
        self.selected_policy = self.selectPolicy()
                
    def categoricalDraw(self,probs):
        
        z = random.random()
        cum_prob = 0.0
        for i in range(len(probs)):
            prob = probs[i]
            cum_prob += prob
            if cum_prob > z:
                return i
        return len(probs) - 1
 
    def selectPolicy(self):
        
        total_exp3weight = sum(self.exp3weights)
        probs = [0.0 for i in range(self.n_policies)]
        for arm in range(self.n_policies):
            probs[arm] = (1 - self.exp3gamma) * (self.exp3weights[arm] / total_exp3weight)
            probs[arm] = probs[arm] + (self.exp3gamma) * (1.0 / float(self.n_policies))
        return self.categoricalDraw(probs)
    
    def update(self, state, action, nextState, reward, positiveReward, negativeReward):
        
        feats = self.featExtractor.getFeatures(state, action)

        cum_R = reward + self.discount * self.computeValueFromQValues(nextState)
        cum_nR = self.n2 * negativeReward + self.discount * self.computeValueFromNegQValues(nextState)
        cum_pR = self.p2 * positiveReward + self.discount * self.computeValueFromPosQValues(nextState)
        difference_q = cum_R - self.getQValue(state, action)
        difference_qn = cum_nR - self.getNegQValue(state, action)
        difference_qp = cum_pR - self.getPosQValue(state, action)
        for i in feats:
            self.weights[i] += self.alpha * difference_q * feats[i]
            self.weights_neg[i] = self.n1 * self.weights_neg[i] + self.alpha * difference_qn * feats[i]
            self.weights_pos[i] = self.p1 * self.weights_pos[i] + self.alpha * difference_qp * feats[i]

        self.iterations += 1;

        total_exp3weight = sum(self.exp3weights)
        probs = [0.0 for i in range(self.n_policies)]
        for arm in range(self.n_policies):
            probs[arm] = (1 - self.exp3gamma) * (self.exp3weights[arm] / total_exp3weight)
            probs[arm] = probs[arm] + (self.exp3gamma) * (1.0 / float(self.n_policies))
          
        exp3reward = (reward - Rmin) / (Rmax - Rmin)
        x = exp3reward / probs[self.selected_policy]
        growth_factor = math.exp((self.exp3gamma / self.n_policies) * x)
        self.exp3weights[self.selected_policy] = self.exp3weights[self.selected_policy] * growth_factor

#############################################################################
# BL ADDED: CLASS WRITTEN FOR Sliding Window UCB-based SPLIT Q-LEARN AGENT  #
#############################################################################

class SWUCBQAgent(MSQL):
    """
       SWUCB-based SPLIT Q-Learning Agent
    """
    def __init__(self,extractor='IdentityExtractor',**args):
        
        self.featExtractor = util.lookup(extractor, globals())()
        MSQL.__init__(self, **args)

        # for SWUCB
        self.last_rewards = np.zeros(self.ucbwindow)  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.last_choices = np.full(self.ucbwindow, -1)  #: Keep in memory the times where each arm was last seen.
        self.selected_policy = self.selectPolicy()
         
    def selectPolicy(self):
  
        armIndex = [0.0 for i in range(self.n_policies)]
        for arm in range(self.n_policies):
            last_pulls_of_this_arm = np.count_nonzero(self.last_choices == arm)
            if last_pulls_of_this_arm < 1:
                armIndex[arm] = float('+inf')
            else:
                armIndex[arm] = (np.sum(self.last_rewards[self.last_choices == arm]) / last_pulls_of_this_arm) + sqrt((self.ucbalpha * log(min(self.iterations, self.ucbwindow))) / last_pulls_of_this_arm)
        return np.argmax(armIndex)

    def update(self, state, action, nextState, reward, positiveReward, negativeReward):
        feats = self.featExtractor.getFeatures(state, action)

        cum_R = reward + self.discount * self.computeValueFromQValues(nextState)
        cum_nR = self.n2 * negativeReward + self.discount * self.computeValueFromNegQValues(nextState)
        cum_pR = self.p2 * positiveReward + self.discount * self.computeValueFromPosQValues(nextState)
        difference_q = cum_R - self.getQValue(state, action)
        difference_qn = cum_nR - self.getNegQValue(state, action)
        difference_qp = cum_pR - self.getPosQValue(state, action)
        for i in feats:
            self.weights[i] += self.alpha * difference_q * feats[i]
            self.weights_neg[i] = self.n1 * self.weights_neg[i] + self.alpha * difference_qn * feats[i]
            self.weights_pos[i] = self.p1 * self.weights_pos[i] + self.alpha * difference_qp * feats[i]

        ucbreward = (reward - Rmin) / (Rmax - Rmin)
        now = self.iterations % self.ucbwindow
        self.last_choices[now] = self.selected_policy
        self.last_rewards[now] = ucbreward
        
        self.iterations += 1;
        


