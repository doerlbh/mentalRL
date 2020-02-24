from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from scipy.stats import ttest_ind

import statsmodels.api as sm


import os
import sys

# load a prespecified scenario

def load_IGT(scheme_id = 1,prefix=''):
    # Fridberg et al., 2009
    if scheme_id == 1: 
        def reward_from_A(N):
            p = np.random.random()
            if (p < .1):
                return [100,np.random.normal(-150, 0)]
            elif (p < .2):
                return [100,np.random.normal(-200, 0)]
            elif (p < .3):
                return [100,np.random.normal(-250, 0)]           
            elif (p < .4):
                return [100,np.random.normal(-300, 0)]
            elif (p < .5):
                return [100,np.random.normal(-350, 0)]
            else:
                return [100,np.random.normal(0, 0)]
                
        def reward_from_B(N):
            p = np.random.random()
            if (p < .1):
                return [100,np.random.normal(-1250, 0)]
            else:
                return [100,np.random.normal(0, 0)]

        def reward_from_C(N):
            p = np.random.random()
            if (p < .1):
                return [50,np.random.normal(-25, 0)]
            elif (p < .2):
                return [50,np.random.normal(-75, 0)]
            elif (p < .5):
                return [50,np.random.normal(-50, 0)]
            else:
                return [50,np.random.normal(0, 0)]
            
        def reward_from_D(N):
            p = np.random.random()
            if (p < .1):
                return [50,np.random.normal(-250, 0)]
            else:
                return [50,np.random.normal(0, 0)]
    
        fig_name = './figures/'+prefix+'IGT_1'

    elif scheme_id == 2: 
        def reward_from_A(N):
            p = np.random.random()
            if (p < .1):
                return [100,np.random.normal(-150, 0)]
            elif (p < .2):
                return [100,np.random.normal(-200, 0)]
            elif (p < .3):
                return [100,np.random.normal(-250, 0)]           
            elif (p < .4):
                return [100,np.random.normal(-300, 0)]
            elif (p < .5):
                return [100,np.random.normal(-350, 0)]
            else:
                return [100,np.random.normal(0, 0)]
                
        def reward_from_B(N):
            p = np.random.random()
            if (p < .1):
                return [100,np.random.normal(-1250, 0)]
            else:
                return [100,np.random.normal(0, 0)]

        def reward_from_C(N):
            p = np.random.random()
            if (p < .5):
                return [50,np.random.normal(-50, 0)]
            else:
                return [50,np.random.normal(0, 0)]
            
        def reward_from_D(N):
            p = np.random.random()
            if (p < .1):
                return [50,np.random.normal(-250, 0)]
            else:
                return [50,np.random.normal(0, 0)]
    
        fig_name = './figures/'+prefix+'IGT_2'
                
    return fig_name,reward_from_A,reward_from_B,reward_from_C,reward_from_D


def load_scenario(scenario_id = 1,is_flipped=False,keyword='MDP',prefix=''):
    if scenario_id == 1:
        def reward_from_B():
            return np.random.normal(-0.5, 10)
        label_B = 'left: N(-0.5,10)'

        def reward_from_C():
            return np.random.normal(0, 1)
        label_C = 'right: N(0,1)'
        fig_name = './figures/'+prefix+''+keyword+'_1'

    elif scenario_id == 2:
        def reward_from_B():
            return np.random.normal(-0.5, 1)
        label_B = 'left: N(-0.5,1)'

        def reward_from_C():
            return np.random.normal(0, 1)
        label_C = 'right: N(0,1)'
        fig_name = './figures/'+prefix+''+keyword+'_2'

    elif scenario_id == 3:
        def reward_from_B():
            return np.random.normal(-0.5, 10)
        label_B = 'left: N(-0.5,10)'

        def reward_from_C():
            return np.random.normal(0, 10)
        label_C = 'right: N(0,10)'
        fig_name = './figures/'+prefix+''+keyword+'_3' 

    elif scenario_id == 4:
        def reward_from_B():
            return np.random.normal(-5, 1)
        label_B = 'left: N(-5,1)'

        def reward_from_C():
            return np.random.normal(0, 1)
        label_C = 'right: N(0,1)'
        fig_name = './figures/'+prefix+''+keyword+'_4' 

    elif scenario_id == 5:
        def reward_from_B():
            return np.random.normal(-5, 10)
        label_B = 'left: N(-5,10)'

        def reward_from_C():
            return np.random.normal(0, 10)
        label_C = 'right: N(0,10)'
        fig_name = './figures/'+prefix+''+keyword+'_5'

    elif scenario_id == 6:
        def reward_from_B():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-10, 10)
            else:
                return np.random.normal(5, 10)
        label_B = 'left: bimodal N(-10,10):N(5,10) = 1:1'

        def reward_from_C():            return np.random.normal(0, 10)
        label_C = 'right: N(0,10)'
        fig_name = './figures/'+prefix+''+keyword+'_6'

    elif scenario_id == 7:
        def reward_from_B():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-10, 10)
            else:
                return np.random.normal(5, 10)
        label_B = 'left: bimodal N(-10,10):N(5,10) = 1:1'

        def reward_from_C():
            return np.random.normal(0, 1)
        label_C = 'right: N(0,1)'
        fig_name = './figures/'+prefix+''+keyword+'_7'

    elif scenario_id == 8:
        def reward_from_B():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-10, 1)
            else:
                return np.random.normal(5, 1)
        label_B = 'left: bimodal N(-10,1):N(5,1) = 1:1'

        def reward_from_C():
            return np.random.normal(0, 1)
        label_C = 'right: N(0,1)'
        fig_name = './figures/'+prefix+''+keyword+'_8'

    elif scenario_id == 9:
        def reward_from_B():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-10, 1)
            else:
                return np.random.normal(5, 1)

        label_B = 'left: bimodal N(-10,1):N(5,1) = 1:1'

        def reward_from_C():
            return np.random.normal(0, 10)
        label_C = 'right: N(0,10)'
        fig_name = './figures/'+prefix+''+keyword+'_9'

    elif scenario_id == 10:
        def reward_from_B():
            p = np.random.random()
            if (p < .75):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(5, 1)
        label_B = 'left: bimodal N(-5,1):N(5,1) = 3:1'

        def reward_from_C():
            return np.random.normal(0, 10)
        label_C = 'right: N(0,10)'
        fig_name = './figures/'+prefix+''+keyword+'_10'

    elif scenario_id == 11:
        def reward_from_B():
            p = np.random.random()
            if (p < .75):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(5, 1)
        label_B = 'left: bimodal N(-5,1):N(5,1) = 3:1'

        def reward_from_C():
            p = np.random.random()
            return np.random.normal(0, 1)
        label_C = 'right: N(0,1)'
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 12:
        def reward_from_B():
            p = np.random.random()
            if (p < .75):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(5, 1)
        label_B = 'left: bimodal N(-5,1):N(5,1) = 3:1'

        def reward_from_C():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(5, 1)
        label_C = 'right: bimodal N(-5,1):N(5,1) = 1:1'
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 13:
        def reward_from_B():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-10, 1)
            else:
                return np.random.normal(5, 1)
        label_B = 'left: bimodal N(-10,1):N(5,1) = 1:1'

        def reward_from_C():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(5, 1)
        label_C = 'right: bimodal N(-5,1):N(5,1) = 1:1'
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 14:
        def reward_from_B():
           return np.random.normal(-0.5, 10)
        label_B = 'left: N(-0.5,10)'

        def reward_from_C():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(5, 1)
        label_C = 'right: bimodal N(-5,1):N(5,1) = 1:1'
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 15:
        def reward_from_B():
            return np.random.normal(-0.5, 1)
        label_B = 'left: N(-0.5,1)'

        def reward_from_C():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(5, 1)
        label_C = 'right: bimodal N(-5,1):N(5,1) = 1:1'
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 16:
        def reward_from_B():
            return np.random.normal(-5, 1)
        label_B = 'left: N(-5,1)'

        def reward_from_C():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(5, 1)
        label_C = 'right: bimodal N(-5,1):N(5,1) = 1:1'
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 17:
        def reward_from_B():
            return np.random.normal(-5, 10)
        label_B = 'left: N(-5,10)'

        def reward_from_C():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(5, 1)
        label_C = 'right: bimodal N(-5,1):N(5,1) = 1:1'
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 18:
        def reward_from_B():
            p = np.random.random()
            if (p < .5):
                return np.random.normal(-10, 1)
            else:
                return np.random.normal(5, 1)
        label_B = 'left: bimodal N(-10,1):N(5,1) = 1:1'

        def reward_from_C():
            p = np.random.random()
            if (p < .75):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(15, 1)
        label_C = 'right: bimodal N(-5,1):N(15,1) = 3:1'
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 19:
        def reward_from_B():
            p = np.random.random()
            if (p < .25):
                return np.random.normal(-20, 1)
            else:
                return np.random.normal(5, 1)
        label_B = 'left: bimodal N(-20,1):N(5,1) = 1:3'

        def reward_from_C():
            p = np.random.random()
            if (p < .75):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(15, 1)
        label_C = 'right: bimodal N(-5,1):N(15,1) = 3:1'
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 20:
        def reward_from_B():
            p = np.random.random()
            if (p < .999):
                return np.random.normal(-1, 1)
            else:
                return np.random.normal(100, 1)
    #     return np.random.normal(-0.5, 10)
        label_B = 'left: bimodal N(-1,1):N(100,1) = 999:1'

        def reward_from_C():
            p = np.random.random()
            if (p < .75):
                return np.random.normal(-5, 1)
            else:
                return np.random.normal(15, 1)
        label_C = 'right: bimodal N(-5,1):N(15,1) = 3:1'
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1000:

        pfB1 = 0.04
        pfB2 = 0.42
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = -0.36
        p1sB = 0
        p2mB = -0.11
        p2sB = 0.06
        p3mB = -0.39
        p3sB = 0.05
        
#         pfC1 = np.random.random()
#         pfC2 = np.random.random()
        pfC1 = 0.03
        pfC2 = 0.08
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = 0.1
        p1sC = 0.02
        p2mC = -0.38
        p2sC = 0.05
        p3mC = -0.06
        p3sC = 0.06
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(1-pfB2)*p3mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(1-pfC2)*p3mC:
            pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC = pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            else:
                return np.random.normal(p3mB, p3sB)
        label_B = 'left: trimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(1-pfB2)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            else:
                return np.random.normal(p3mC, p3sC)
        label_C = 'right: trimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(1-pfC2)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

        pfB1 = 0.04
        pfB2 = 0.42
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = -0.36
        p1sB = 0
        p2mB = -0.11
        p2sB = 0.06
        p3mB = -0.39
        p3sB = 0.05
        
#         pfC1 = np.random.random()
#         pfC2 = np.random.random()
        pfC1 = 0.03
        pfC2 = 0.08
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = 0.1
        p1sC = 0.02
        p2mC = -0.38
        p2sC = 0.05
        p3mC = -0.06
        p3sC = 0.06
        
    elif scenario_id == 1001:

        pfB1 = 0.04
        pfB2 = 0.42
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = -36
        p1sB = 0
        p2mB = -11
        p2sB = 6
        p3mB = -39
        p3sB = 5
        
#         pfC1 = np.random.random()
#         pfC2 = np.random.random()
        pfC1 = 0.03
        pfC2 = 0.08
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = 10
        p1sC = 2
        p2mC = -38
        p2sC = 5
        p3mC = -6
        p3sC = 6
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(1-pfB2)*p3mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(1-pfC2)*p3mC:
            pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC = pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            else:
                return np.random.normal(p3mB, p3sB)
        label_B = 'left: trimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(1-pfB2)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            else:
                return np.random.normal(p3mC, p3sC)
        label_C = 'right: trimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(1-pfC2)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1002:

        pfB1 = 0.04
        pfB2 = 0.42
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = -36
        p1sB = 0
        p2mB = 0
        p2sB = 6
        p3mB = -50
        p3sB = 5
        
#         pfC1 = np.random.random()
#         pfC2 = np.random.random()
        pfC1 = 0.03
        pfC2 = 0.08
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = 10
        p1sC = 2
        p2mC = -38
        p2sC = 5
        p3mC = -6
        p3sC = 6
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(1-pfB2)*p3mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(1-pfC2)*p3mC:
            pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC = pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            else:
                return np.random.normal(p3mB, p3sB)
        label_B = 'left: trimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(1-pfB2)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            else:
                return np.random.normal(p3mC, p3sC)
        label_C = 'right: trimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(1-pfC2)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1003:

        pfB1 = 0.04
        pfB2 = 0.42
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = -36
        p1sB = 0
        p2mB = -11
        p2sB = 6
        p3mB = -39
        p3sB = 5
        
#         pfC1 = np.random.random()
#         pfC2 = np.random.random()
        pfC1 = 0.03
        pfC2 = 0.08
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = 10
        p1sC = 2
        p2mC = -50
        p2sC = 5
        p3mC = 0
        p3sC = 6
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(1-pfB2)*p3mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(1-pfC2)*p3mC:
            pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC = pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            else:
                return np.random.normal(p3mB, p3sB)
        label_B = 'left: trimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(1-pfB2)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            else:
                return np.random.normal(p3mC, p3sC)
        label_C = 'right: trimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(1-pfC2)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1004:

        pfB1 = 0.04
        pfB2 = 0.42
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = -36
        p1sB = 0
        p2mB = -20
        p2sB = 6
        p3mB = -30
        p3sB = 5
        
        pfC1 = 0.03
        pfC2 = 0.08
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = 10
        p1sC = 2
        p2mC = -38
        p2sC = 5
        p3mC = -6
        p3sC = 6
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(1-pfB2)*p3mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(1-pfC2)*p3mC:
            pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC = pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            else:
                return np.random.normal(p3mB, p3sB)
        label_B = 'left: trimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(1-pfB2)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            else:
                return np.random.normal(p3mC, p3sC)
        label_C = 'right: trimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(1-pfC2)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1005:

        pfB1 = 0.0
        pfB2 = 0.4
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = -36
        p1sB = 0
        p2mB = -11
        p2sB = 6
        p3mB = -39
        p3sB = 5
        
#         pfC1 = np.random.random()
#         pfC2 = np.random.random()
        pfC1 = 0.0
        pfC2 = 0.07
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = 10
        p1sC = 2
        p2mC = -38
        p2sC = 5
        p3mC = -6
        p3sC = 6
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(1-pfB2)*p3mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(1-pfC2)*p3mC:
            pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC = pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            else:
                return np.random.normal(p3mB, p3sB)
        label_B = 'left: trimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(1-pfB2)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            else:
                return np.random.normal(p3mC, p3sC)
        label_C = 'right: trimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(1-pfC2)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)
        
    elif scenario_id == 1006:

        pfB1 = 0.04
        pfB2 = 0.42
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = -36
        p1sB = 0
        p2mB = -11
        p2sB = 12
        p3mB = -39
        p3sB = 10
        
#         pfC1 = np.random.random()
#         pfC2 = np.random.random()
        pfC1 = 0.03
        pfC2 = 0.08
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = 10
        p1sC = 2
        p2mC = -38
        p2sC = 5
        p3mC = -6
        p3sC = 6
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(1-pfB2)*p3mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(1-pfC2)*p3mC:
            pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC = pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            else:
                return np.random.normal(p3mB, p3sB)
        label_B = 'left: trimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(1-pfB2)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            else:
                return np.random.normal(p3mC, p3sC)
        label_C = 'right: trimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(1-pfC2)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1007:

        pfB1 = 0.04
        pfB2 = 0.42
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = -36
        p1sB = 0
        p2mB = -11
        p2sB = 3
        p3mB = -39
        p3sB = 2
        
#         pfC1 = np.random.random()
#         pfC2 = np.random.random()
        pfC1 = 0.03
        pfC2 = 0.08
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = 10
        p1sC = 2
        p2mC = -38
        p2sC = 5
        p3mC = -6
        p3sC = 6
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(1-pfB2)*p3mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(1-pfC2)*p3mC:
            pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC = pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            else:
                return np.random.normal(p3mB, p3sB)
        label_B = 'left: trimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(1-pfB2)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            else:
                return np.random.normal(p3mC, p3sC)
        label_C = 'right: trimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(1-pfC2)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)
                                
    elif scenario_id == 1008:

        pfB1 = 0.04
        pfB2 = 0.42
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = -36
        p1sB = 0
        p2mB = -11
        p2sB = 6
        p3mB = -39
        p3sB = 5
        
#         pfC1 = np.random.random()
#         pfC2 = np.random.random()
        pfC1 = 0.03
        pfC2 = 0.08
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = 10
        p1sC = 4
        p2mC = -38
        p2sC = 10
        p3mC = -6
        p3sC = 12
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(1-pfB2)*p3mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(1-pfC2)*p3mC:
            pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC = pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            else:
                return np.random.normal(p3mB, p3sB)
        label_B = 'left: trimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(1-pfB2)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            else:
                return np.random.normal(p3mC, p3sC)
        label_C = 'right: trimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(1-pfC2)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)
                                
    elif scenario_id == 1009:

        pfB = 0.42
        p1mB = -11
        p1sB = 6
        p2mB = -39
        p2sB = 5
        
        pfC = 0.07
        p1mC = -38
        p1sC = 10
        p2mC = -6
        p2sC = 12
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1010:
        
        pfB = 0.42
        p1mB = -11
        p1sB = 6
        p2mB = -39
        p2sB = 5
        
        pfC = 0.07
        p1mC = -50
        p1sC = 10
        p2mC = 0
        p2sC = 12
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)
                                                                
    elif scenario_id == 1011:

        pfB = 0.3
        p1mB = -11
        p1sB = 6
        p2mB = -39
        p2sB = 5
        
        pfC = 0.07
        p1mC = -38
        p1sC = 10
        p2mC = -6
        p2sC = 12
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)
        
    elif scenario_id == 1012:

        pfB = 0.7
        p1mB = -11
        p1sB = 6
        p2mB = -39
        p2sB = 5
        
        pfC = 0.07
        p1mC = -38
        p1sC = 10
        p2mC = -6
        p2sC = 12
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1013:

        pfB = 0.3
        p1mB = -11
        p1sB = 6
        p2mB = -39
        p2sB = 5
        
        pfC = 0.3
        p1mC = -38
        p1sC = 10
        p2mC = -6
        p2sC = 12
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1014:

        pfB = 0.7
        p1mB = -11
        p1sB = 6
        p2mB = -39
        p2sB = 5
        
        pfC = 0.3
        p1mC = -38
        p1sC = 10
        p2mC = -6
        p2sC = 12
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1015:

        pfB = 0.7
        p1mB = -11
        p1sB = 6
        p2mB = -39
        p2sB = 5
        
        pfC = 0.3
        p1mC = -38
        p1sC = 5
        p2mC = -6
        p2sC = 6
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)
    
    elif scenario_id == 1016:

        pfB = 0.7
        p1mB = -11
        p1sB = 6
        p2mB = -39
        p2sB = 5
        
        pfC = 0.7
        p1mC = -38
        p1sC = 10
        p2mC = -6
        p2sC = 12
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)
    
    elif scenario_id == 1017:

        pfB = 0.7
        p1mB = -11
        p1sB = 6
        p2mB = -39
        p2sB = 5
        
        pfC = 0.7
        p1mC = -38
        p1sC = 5
        p2mC = -6
        p2sC = 6
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)                                                                        
                                                                        
    elif scenario_id == 1018:

        pfB = 0.3
        p1mB = -11
        p1sB = 6
        p2mB = -39
        p2sB = 5
        
        pfC = 0.3
        p1mC = -38
        p1sC = 5
        p2mC = -6
        p2sC = 6
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)                                                                        

    elif scenario_id == 1019:

        pfB = 0.7
        p1mB = -11
        p1sB = 12
        p2mB = -39
        p2sB = 10
        
        pfC = 0.3
        p1mC = -38
        p1sC = 10
        p2mC = -6
        p2sC = 12
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id == 1020:

        pfB = 0.7
        p1mB = 6
        p1sB = 6
        p2mB = -22
        p2sB = 5
        
        pfC = 0.7
        p1mC = 11
        p1sC = 12
        p2mC = -21
        p2sC = 10
        
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)
        

    elif scenario_id < 121:

        pfB = np.random.random()
        p1mB = np.random.randint(-100,100)
        p1sB = np.random.randint(0,20)
        p2mB = np.random.randint(-100,100)
        p2sB = np.random.randint(0,20)
          
        pfC = np.random.random()
        p1mC = np.random.randint(-100,100)
        p1sC = np.random.randint(0,20)
        p2mC = np.random.randint(-100,100)
        p2sC = np.random.randint(0,20)
                    
        if pfB*p1mB+(1-pfB)*p2mB > pfC*p1mC+(1-pfC)*p2mC:
            pfB,p1mB,p1sB,p2mB,p2sB,pfC,p1mC,p1sC,p2mC,p2sC = pfC,p1mC,p1sC,p2mC,p2sC,pfB,p1mB,p1sB,p2mB,p2sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB):
                return np.random.normal(p1mB, p1sB)
            else:
                return np.random.normal(p2mB, p2sB)
        label_B = 'left: bimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+') = {0:.2f}'.format(pfB)+':{0:.2f}'.format(1-pfB)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC):
                return np.random.normal(p1mC, p1sC)
            else:
                return np.random.normal(p2mC, p2sC)
        label_C = 'right: bimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+') = {0:.2f}'.format(pfC)+':{0:.2f}'.format(1-pfC)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id < 221:

        pfB1 = np.random.random()
        pfB2 = np.random.random()
        if pfB1 > pfB2: pfB1,pfB2 = pfB2,pfB1
        p1mB = np.random.randint(-100,100)
        p1sB = np.random.randint(0,50)
        p2mB = np.random.randint(-100,100)
        p2sB = np.random.randint(0,50)
        p3mB = np.random.randint(-100,100)
        p3sB = np.random.randint(0,50)
        
        pfC1 = np.random.random()
        pfC2 = np.random.random()
        if pfC1 > pfC2: pfC1,pfC2 = pfC2,pfC1
        p1mC = np.random.randint(-100,100)
        p1sC = np.random.randint(0,50)
        p2mC = np.random.randint(-100,100)
        p2sC = np.random.randint(0,50)
        p3mC = np.random.randint(-100,100)
        p3sC = np.random.randint(0,50)
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(1-pfB2)*p3mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(1-pfC2)*p3mC:
            pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC = pfC1,pfC2,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,pfB1,pfB2,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            else:
                return np.random.normal(p3mB, p3sB)
        label_B = 'left: trimodal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(1-pfB2)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            else:
                return np.random.normal(p3mC, p3sC)
        label_C = 'right: trimodal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(1-pfC2)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)

    elif scenario_id < 321:

        pfB1 = np.random.random()
        pfB2 = np.random.random()
        pfB3 = np.random.random()
        pfB = [pfB1,pfB2,pfB3]
        pfB.sort()
        pfB1,pfB2,pfB3 = pfB[0],pfB[1],pfB[2]
        p1mB = np.random.randint(-100,100)
        p1sB = np.random.randint(0,50)
        p2mB = np.random.randint(-100,100)
        p2sB = np.random.randint(0,50)
        p3mB = np.random.randint(-100,100)
        p3sB = np.random.randint(0,50)
        p4mB = np.random.randint(-100,100)
        p4sB = np.random.randint(0,50)
        
        pfC1 = np.random.random()
        pfC2 = np.random.random()
        pfC3 = np.random.random()
        pfC = [pfC1,pfC2,pfC3]
        pfC.sort()
        pfC1,pfC2,pfC3 = pfC[0],pfC[1],pfC[2]
        p1mC = np.random.randint(-100,100)
        p1sC = np.random.randint(0,50)
        p2mC = np.random.randint(-100,100)
        p2sC = np.random.randint(0,50)
        p3mC = np.random.randint(-100,100)
        p3sC = np.random.randint(0,50)
        p4mC = np.random.randint(-100,100)
        p4sC = np.random.randint(0,50)
        
        if pfB1*p1mB+(pfB2-pfB1)*p2mB+(pfB3-pfB2)*p3mB+(1-pfB3)*p4mB > pfC1*p1mC+(pfC2-pfC1)*p2mC+(pfC3-pfC2)*p3mC+(1-pfC3)*p4mC:
            pfB1,pfB2,pfB3,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,p4mB,p4sB,pfC1,pfC2,pfC3,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,p4mC,p4sC = pfC1,pfC2,pfC3,p1mC,p1sC,p2mC,p2sC,p3mC,p3sC,p4mC,p4sC,pfB1,pfB2,pfB3,p1mB,p1sB,p2mB,p2sB,p3mB,p3sB,p4mB,p4sB
        
        def reward_from_B():
            p = np.random.random()
            if (p < pfB1):
                return np.random.normal(p1mB, p1sB)
            elif (p < pfB2):
                return np.random.normal(p2mB, p2sB)
            elif (p < pfB3):
                return np.random.normal(p3mB, p3sB)
            else:
                return np.random.normal(p4mB, p4sB)
        label_B = 'left: 4-modal N('+str(p1mB)+','+str(p1sB)+'):N('+str(p2mB)+','+str(p2sB)+'):N('+str(p3mB)+','+str(p3sB)+'):N('+str(p4mB)+','+str(p4sB)+') = {0:.2f}'.format(pfB1)+':{0:.2f}'.format(pfB2-pfB1)+':{0:.2f}'.format(pfB3-pfB2)+':{0:.2f}'.format(1-pfB3)

        def reward_from_C():
            p = np.random.random()
            if (p < pfC1):
                return np.random.normal(p1mC, p1sC)
            elif (p < pfC2):
                return np.random.normal(p2mC, p2sC)
            elif (p < pfC3):
                return np.random.normal(p3mC, p3sC)
            else:
                return np.random.normal(p4mC, p4sC)
        label_C = 'right: 4-modal N('+str(p1mC)+','+str(p1sC)+'):N('+str(p2mC)+','+str(p2sC)+'):N('+str(p3mC)+','+str(p3sC)+'):N('+str(p4mC)+','+str(p4sC)+') = {0:.2f}'.format(pfC1)+':{0:.2f}'.format(pfC2-pfC1)+':{0:.2f}'.format(pfC3-pfC2)+':{0:.2f}'.format(1-pfC3)

        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)
        
    if is_flipped:
        reward_from_C, reward_from_B, label_C, label_B = reward_from_B, reward_from_C, label_B, label_C 
        fig_name = './figures/'+prefix+''+keyword+'_'+str(scenario_id)+'_flipped'
    fig_name_flipped = './figures/'+prefix+''+keyword+'_'+str(scenario_id)+'_flipped'
    kld,crossentropy,sB,sC,sdiff = get_dist_stats(reward_from_B,reward_from_C)
    return label_B,label_C,fig_name,fig_name_flipped,reward_from_B,reward_from_C,kld,crossentropy,sB,sC,sdiff

def get_dist_stats(reward_from_B,reward_from_C):
    dprime,crossentropy = 1,1

    rB,rC = [],[]
    for i in range(5000):
        rB.append(reward_from_B())
        rC.append(reward_from_C())

    kdeB = sm.nonparametric.KDEUnivariate(rB)
    kdeB.fit() # Estimate the densities
    kdeC = sm.nonparametric.KDEUnivariate(rC)
    kdeC.fit() # Estimate the densities

#     return entropy(kdeB.density,kdeC.density), cross_entropy(kdeB.density,kdeC.density)
    return entropy(kdeB.density,kdeC.density), cross_entropy(kdeB.density,kdeC.density),np.std(rB),np.std(rC),np.std(rB)-np.std(rC)

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def collateComparison(nameA,nameB,scenario,scoreA,scoreB,sB,sC,sdiff,crossentropy):
    A = np.array(scoreA)
    B = np.array(scoreB)
    s = np.array(scenario)
    sB = np.array(sB)
    sC = np.array(sC)
    sdiff = np.array(sdiff)
    crossentropy = np.array(crossentropy)

    AoB = sum(A>B)
    BoA = sum(B>A)

    caseAoB = s[A>B]
    caseBoA = s[B>A]
    
    scoreAm,scoreAs = np.mean(A),np.std(A)
    scoreBm,scoreBs = np.mean(B),np.std(B)
    
    stat, scorep = ttest_ind(A, B)

    sBAoB = sB[A>B]
    sCAoB = sC[A>B]
    sdiffAoB = sdiff[A>B]
    crossentropyAoB = crossentropy[A>B]

    sBBoA = sB[A<B]
    sCBoA = sC[A<B]
    sdiffBoA = sdiff[A<B]
    crossentropyBoA = crossentropy[A<B]

    stat, sBp = ttest_ind(sBAoB, sBBoA)
    stat, sCp = ttest_ind(sCAoB, sCBoA)
    stat, sdiffp = ttest_ind(sdiffAoB, sdiffBoA)
    stat, crossentropyp = ttest_ind(crossentropyAoB, crossentropyBoA)

    sBAoBm,sBAoBs = np.mean(sBAoB),np.std(sBAoB)
    sCAoBm,sCAoBs = np.mean(sCAoB),np.std(sCAoB)
    sdiffAoBm,sdiffAoBs = np.mean(sdiffAoB),np.std(sdiffAoB)
    crossentropyAoBm,crossentropyAoBs = np.mean(crossentropyAoB),np.std(crossentropyAoB)

    sBBoAm,sBBoAs = np.mean(sBBoA),np.std(sBBoA)
    sCBoAm,sCBoAs = np.mean(sCBoA),np.std(sCBoA)
    sdiffBoAm,sdiffBoAs = np.mean(sdiffBoA),np.std(sdiffBoA)
    crossentropyBoAm,crossentropyBoAs = np.mean(crossentropyBoA),np.std(crossentropyBoA)

    print('final scores for ', nameA, 'and', nameB, ':')
    print(scoreAm, '+/-', scoreAs, ' vs. ',scoreBm,'+/-',scoreBs, ', p=',scorep)
    print('\n')
    print('Wins: ', nameA,':', nameB, ' = ', str(AoB), ':',str(BoA),'\n')
    print('Cases ', nameA, ' wins: ', caseAoB)
    print('Cases ', nameB, ' wins: ', caseBoA)
    print('\n')
    print('std of B for the winning cases for ', nameA, 'and', nameB, ':')
    print(sBAoBm, '+/-', sBAoBs, ' vs. ',sBBoAm,'+/-',sBBoAs, ', p=',sBp)
    print('std of C for the winning cases for ', nameA, 'and', nameB, ':')
    print(sCAoBm, '+/-', sCAoBs, ' vs. ',sCBoAm,'+/-',sCBoAs, ', p=',sCp)
    print('stdB-stdC for the winning cases for ', nameA, 'and', nameB, ':')
    print(sdiffAoBm, '+/-', sdiffAoBs, ' vs. ',sdiffBoAm,'+/-',sdiffBoAs, ', p=',sdiffp)
    print('crossentropy for the winning cases for ', nameA, 'and', nameB, ':')
    print(crossentropyAoBm, '+/-', crossentropyAoBs, ' vs. ',crossentropyBoAm,'+/-',crossentropyBoAs, ', p=',crossentropyp)

    return AoB,BoA,scoreAm,scoreAs,scoreBm,scoreBs,scorep,caseAoB,caseBoA,sBAoB,sCAoB,sdiffAoB,crossentropyAoB,sBBoA,sCBoA,sdiffBoA,crossentropyBoA,sBp,sCp,sdiffp,crossentropyp,sBAoBm,sBAoBs,sCAoBm,sCAoBs,sdiffAoBm,sdiffAoBs,crossentropyAoBm,crossentropyAoBs,sBBoAm,sBBoAs,sBBoAm,sBBoAs,sCBoAm,sCBoAs,sdiffBoAm,sdiffBoAs,crossentropyBoAm,crossentropyBoAs

# Welford online computation of mean and std
# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1 
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)

# retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)