"""
"""
from scipy.spatial.distance import cdist 
import numpy as np 

class RBF:
    """
        A stationary kernel also known as the squared exponential kernel
        Parameterized by a  length scale 
            k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)


        Parameters
        ----------

        length_scale: 

    """
    def __init__(self, length_scale = 1.0):
        """

        length_sca

        """
        self.length_scale  = length_scale 

    def _validate_length_scale(self, X,length_scale):
        """
        """
        length_scale = np.squeeze(length_scale).astype(float)
        if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
            raise ValueError("Anisotropic kernel must have the same number of "
                             "dimensions as data (%d!=%d)"
                             % (length_scale.shape[0], X.shape[1]))
        return length_scale

    def __call__(self,X,Y = None):
        """Returns the kernel k(X,Y) 

        Parameters
        ----------

        X: array with shape (n_samples_X, n_features)

        Y: array, shape (n_samples_Y, n_features )  
           if None, k(X,X ) is evaluated .

        Returns
        -------
 
        K : array, shape (n_samples_X, n_samples_Y )
        """

        # Ensure we have a 2d shape numpy array for iteration puproses 
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        
        length_scale = self._validate_length_scale(X,self.length_scale)

        dists = cdist(X/length_scale, Y/length_scale, metric = 'sqeuclidean')
        K = np.exp(-.5 * dists )

        return K 

class Noise:
    """

    """

    def __init__(self, noise_size):
        self.noise_size = noise_size 

    def __validated_noise_size(self,X,noise_size):
        """
            Ensures that the width is a valid type, run test. 
        """
        return noise_size

    def __call__(self,X,Y = None): 
        if Y is None:
            K = self.noise_size * np.eye(X.shape[0])
            return K
        else:
            return np.zeros((X.shape[0], Y.shape[0]))


