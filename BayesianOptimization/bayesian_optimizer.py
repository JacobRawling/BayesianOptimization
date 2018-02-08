"""
    A Tool to perform Bayesian optimization on a black box function, f, 
    using Gaussina Processes.

    This tool supports 
"""
import numpy as np
import kernels 
import utility_functions as uf
from copy import copy

class Iteration:
    def __init__(self, 
            mean,
            covariance, 
            X,
            f_X,
            u,
            next_x,
            best_u
                ):
        """ 
            A container for an iteration in a bayesian optimizaiton 
        """
        self.mean       = copy(mean)
        self.covariance = copy(covariance)
        self.width      = copy(np.diag(covariance))
        self.X          = copy(X)
        self.f_X        = copy(f_X)
        self.u          = copy(u)

        # We do not know the way in which u is calculated 
        # therefore we also need to store the optimal x and u 
        # values, as found by the utiliy function 
        self.next_x  = next_x
        self.best_u  = best_u

class BayesianOptimizer:
    """
        A generic class to optimize an arbitrary unkown function f in as 
        few evaluation steps as possible


        Parameters
        ----------

        kernel:
            The kernel for the Gaussian processes to be evaluated. 
            There are several options for this:
                * RBF: Radial basis function, also known as squared-exponential kernel 
                * Matern: A generalized RBF
            Alternatively a user can prass a function that returns a correlation matrix
            of dimension 
            where M is the 

        utility: 
            The utility function that decides the direction of the optimization 
            Sometimes called an aquisiton fucntion

        verbostiy: 
            The amount of text this class will output to stdout, default value
            of 1 which is the maximum 
    
    """

    def __init__(self,
            kernel = 'RBF',
            utility = 'expected_improvement',   
            verbosity = 1       
        ):


        # Load the kernel function, ensuring user  defined functions pass a 
        # basic set of tests.
        if callable(kernel):
            assert False, "ERROR: kernel function support not implemented yet!"
        else:
            self.kernel  = kernels.RBF()

        # Similarly load the utility function and ensure that a user define
        # defined function passes the basic behaviourly chaaracteristics
        # of a utlitly function
        if callable(utility):
            assert False, "ERROR: kernel function support not implemented yet!"
        else:
            self.utility  = uf.ExpectedImprovement()

        self.verbosity = verbosity

    def log(self, message):
        """
            Sends a message to standard output stream provided that the verbosity
            of the class is greater than 0 
        """
        if self.verbosity >0:
            print ("[ BA-OPT ] - " + str(message))

    def optimize(self,f,X_test, X_initial,f_X_initial,
         convergence_criteria = 5,
         save_iterations= True):
        """
            

        save_iterations:
            A flag to save and return all iterations of optimization ran by this tool
        """
        self.log(" Optimizing..." )

        X = X_initial
        f_X = f_X_initial
        iterations = []
        n_iterations = 0
        # We allow for a general convergence test defined by the user 
        # Continue while this test is fales 
        while not self.is_converged(convergence_criteria, n_iterations):
            self.log(" Iteration: " + str(n_iterations))
            n_iterations += 1

            # Evaluate the postier distribution - in particular grab
            # the mean and covariance at each point in X_test
            mean, covariance = self.postierier( 
                        X_test,
                        X_initial,
                        f_X_initial
                    )

            # Evaluate the width at each point in X_test
            # of the postier.
            width = np.diag(covariance) 

            # Using the utility function evaluate the next
            # best point to evaluate 
            u, next_x, best_u = self.utility(X_test,X,f_X, mean,width)

            # Store the details so they are accesible for later
            # drawing and minipulation 
            if save_iterations:
                iterations.append( Iteration(
                        mean,
                        covariance,
                        X,
                        f_X,
                        u,
                        next_x,
                        best_u
                    ))

            self.log(" Evaluating next x: " + str(next_x))
            f_X = vstack((f_X, f(next_x)))
            X.vstack((x, f_X[-1,:]) )

        self.log(" Complete." )
        return iterations 



    def is_converged(self, convergence_criteria, param):
        """ 

        """
        self.log ("convergence_criteria: is integer: " + str(isinstance(convergence_criteria,int)))
        if isinstance(convergence_criteria,int):
            return param >= convergence_criteria
        else:
            return True



    def postierier(self,x,X,f_X):
        """ 
        Evaluates  the postier given a set of measured values and defined kernel function:

        Parameters
        ----------
        x: distibution in x to be tested over 
      
        X: measured X that will be integrated out 
      
        f_X: measured Y associated with each point x    
        """

        k_xX = self.kernel(x,X)
        k_XX = self.kernel(X,X)
        k_XX_inv = np.linalg.inv(k_XX)
        k_xx = self.kernel(x,x)
        k_Xx = self.kernel(X,x)
        
        mean = np.matmul(k_XX_inv,f_X )
        mean = np.matmul( k_xX, mean)

        sigma = np.matmul(k_XX_inv, k_Xx)
        sigma = np.matmul(k_xX, sigma)
        sigma = k_xx - sigma

        return mean, sigma