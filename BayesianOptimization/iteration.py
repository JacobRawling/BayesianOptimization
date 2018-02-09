"""
"""
from copy import copy
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

import numpy as np 

class Iteration:
    def __init__(self, 
            mean,
            covariance, 
            X,
            f_X,
            u,
            next_x,
            best_u,
            x_true
            ):
        """ 
            A container for an iteration in a bayesian optimizaiton 
        """
        self.mean       = copy(mean)
        self.covariance = copy(covariance)
        self.width      = np.atleast_2d(np.diag(covariance)).T
        self.X          = copy(X)
        self.f_X        = copy(f_X)
        self.u          = copy(u)
        self.x_true     = copy(x_true)

        # We do not know the way in which u is calculated 
        # therefore we also need to store the optimal x and u 
        # values, as found by the utiliy function 
        self.next_x  = next_x
        self.best_u  = best_u

    def find_mean_width(self,x_feature,f_feature):
        x = iterations[-1].x_true[:,0]
        m = iterations[-1].mean[:,0]
        x,indicies = np.unique(x,return_index=True)

        means = []
        for i in xrange(len(indicies)-1):
            print "averaging from ",indicies[i]," to", indicies[i+1]
            means.append(np.average(m[indicies[i]:indicies[i+1]]))
        means.append(np.average(m[indicies[-1]:]))


    def draw(self,x_feature = 0, f_feature = 0, X_test = None,f_X_true = None):
        """
        """
        m = self.mean[:,f_feature]
        sig = self.width[:,f_feature]

        # If we are in n > 1 dimensions then we need to
        # reduce the x-axis down to a 
        X = self.X[:,x_feature]
        # Actually want to average 
        f_X = self.f_X[:,f_feature]
        X_true = self.x_true[:,x_feature]
        
        #Creat a blank 
        fig = plt.figure(figsize=(8, 8))

        # Setup a figure with an upper and lower pannel 
        gs = gridspec.GridSpec(2, 1, width_ratios=[1],height_ratios=[3,1]) 
        ax0 = plt.subplot(gs[0])

        # Draw the upwer pannel
        # ax0.fill_between(X_true, m - 2*sig, m+2*sig,alpha=0.5,color='green',label = r'$\pm 2\sigma$')
        # ax0.fill_between(X_true, m - sig, m+sig,alpha=0.5,color='yellow',label = r'$\pm 1\sigma$')


        # if type(f_X_true).__module__ == np.__name__ or isinstance(f_X_true, list):
        #     ax0.plot(X_true,f_X_true,color='red', label='Target', linewidth = 2.0)

        ax0.plot(X_true,m,color = 'black', label='Mean')
        # ax0.plot(X,f_X,'bo', label='Measured')

        ax0.legend()

        # Draw the utility function for the current postier 
        u = self.u[:,f_feature]

        ax1 = plt.subplot(gs[1])
        ax1.fill_between(X_true, 0,u,alpha=0.5,color='blue')
        ax1.plot( [self.next_x[x_feature] ],[self.best_u], 'y*',
                  markersize=24)
        ax1.set_xlabel('x')
        ax1.set_ylabel('U(x)')

        plt.tight_layout()
        plt.show()
        plt.close()
        return fig 
