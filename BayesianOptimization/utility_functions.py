"""
"""
from scipy.spatial.distance import cdist 
from scipy.stats import norm
import numpy as np 

class ExpectedImprovement:
    """
        A stationary kernel also known as the squared exponential kernel
        Parameterized by a  length scale 
            k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)


        Parameters
        ----------

        length_scale: 

    """
    def __init__(self):
        """

        length_sca
        """


    def __call__(self,X_test,X,fX,mean,width):
        """Returns the kernel k(X,Y) 

        Parameters
        ----------

        X:  array with shape (n_samples_X, n_features)

        fX: array with shape (n_samples_X, )

        mean: array 

        width: 

        Y: array, shape (n_samples_Y, n_features )  
           if None, k(X,X ) is evaluated .

        Returns
        -------
 
        K : array, shape (n_samples_X, n_samples_Y )
        """

        # Ensure we have a 2d shape numpy array for iteration puproses 
        f_max = np.max(fX)
        f_min = np.max(fX)
        z = (mean-f_min)

        width = np.atleast_2d(width).T
        u = (mean - f_min)*norm.cdf(z) + width*norm.pdf(z)
        x_max, u_max = X_test[np.argmax(u)],np.max(u)

        return u,x_max,u_max
        
