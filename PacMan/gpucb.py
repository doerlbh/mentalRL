import math
import random

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as spopt
from sklearn import gaussian_process as skgp

from reciprocation.learningstrategies import eval_point


class GPUCB:
    """
    Gaussian Process UCB
    """
    def __init__(self,kernel=None,kappa=1.0,history_window=100,minimizestarts=10,gpstarts=25,fitfreq=10,alpha=1e-10,startmove=None):
        if kernel is None:
            kernel=skgp.kernels.RBF(length_scale=1.0,length_scale_bounds=(.2,100))+skgp.kernels.WhiteKernel(noise_level=1.0)
            self.gp=skgp.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=gpstarts,alpha=alpha)
        else:
            self.gp=skgp.GaussianProcessRegressor(kernel=kernel,alpha=1e-10,n_restarts_optimizer=gpstarts)
        self.kernel=kernel
        self.initkernel=kernel
        self.move=[]
        self.response=[]
        self.kappa=kappa
        self.alpha=alpha
        self.lastmove=None
        self.history_window=history_window
        self.gpstarts=gpstarts
        self.n=0
        self.minimizestarts=minimizestarts
        self.fitfreq=fitfreq
        self.gpparams=None
        self.startmove=startmove

    def reset(self): # TODO Check for bugs in reset/clone
        self.n=0
        self.lastmove=None
        self.move=[]
        self.response=[]
        self.kernel=self.initkernel
        self.gp = skgp.GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.gpstarts, alpha=self.alpha)

    def clone(self):
        result=GPUCB(self.kernel,self.kappa,self.history_window,self.minimizestarts,self.gpstarts,self.fitfreq,self.alpha)
        result.move=[m for m in self.move]
        result.response=[r for r in self.response]
        result.lastmove=self.lastmove
        result.gpparams=self.gpparams
        result.n=self.n
        return result

    def __str__(self):
        return "GPUCB"

    def __repr__(self):
        return str(self)

    def getStatus(self):
        return str(self)

    def getDescription(self):
        return str(self)

    def respond(self,opponentmove):
        if self.lastmove is not None:
            self.update(self.lastmove,opponentmove)
        self.lastmove=self.pickmove(opponentmove)
        return self.lastmove

    def update(self,move,response):
        self.n=self.n+1
        self.move.append(move)
        self.response.append(response)
        self.move=self.move[-self.history_window:]
        self.response=self.response[-self.history_window:]
        if self.n<self.fitfreq or self.n%self.fitfreq==0:
            self.gp.optimizer="fmin_l_bfgs_b"
        else:
            self.gp.optimizer=None
            self.gp.kernel.set_params(**self.gpparams)
        self.gp.fit(np.array(self.move).reshape(-1,1),np.array(self.response).reshape(-1,1))
        if False:
            self.dispGP()
        if self.n < self.fitfreq or self.n % self.fitfreq == 0:
            self.gpparams=self.gp.kernel_.get_params(True)
            del self.gpparams['k1']
            del self.gpparams['k2']

    def pickmove(self,oppmove):
        if self.startmove is not None and oppmove is None:
            return self.startmove
        maxresult=None
        for i in range(self.minimizestarts):
            result=spopt.minimize(fun=lambda x: eval_point(x,self.gp,self.kappa,lambda x:0),x0=(2*random.random()-1,),bounds=np.array(((-1.0,1.0),)),method="L-BFGS-B")
            if maxresult is None or -result.fun[0]>maxresult.fun[0]:
                maxresult=result
        return max(-1.0,min(1.0,maxresult.x[0]))

    def checkpoint(self,x):
        mean, std = self.gp.predict(np.array(x).reshape(-1, 1), return_std=True)
        print "Mean: "+str(mean)
        print "Std: "+str(std)
        print "Kappa: "+str(self.kappa)
        print "Own payoff: "+str(math.sqrt(1-x**2))
        print "Result: "+str(-mean-self.kappa*std-math.sqrt(1-x**2))

    def dispGP(self):
        mean,std=self.gp.predict(np.arange(-1,1,.01).reshape(-1,1),return_std=True)
        eval=[m + self.kappa * s + math.sqrt(1-x**2) for x,m,s in zip(np.arange(-1,1,.01),mean,std)]
        plt.figure(figsize=(16,9))
        plt.plot(np.arange(-1,1,.01),mean)
        plt.plot(np.arange(-1,1,.01),eval)
        plt.plot(np.arange(-1,1,.01),[m+math.sqrt(1-x**2) for x,m in zip(np.arange(-1,1,.01),mean)])
        plt.fill_between(np.arange(-1,1,.01),np.squeeze(mean)-std,np.squeeze(mean)+std,alpha=.1)
        plt.scatter(self.move,self.response,c="red",s=50)
        plt.xlim(-1,1)
        plt.ylim(-2,2)
        plt.show()