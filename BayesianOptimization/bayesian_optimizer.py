"""
    A Tool to perform Bayesian optimization on a black box function, f, 
    using Gaussina Processes.

    This tool supports 
"""
import numpy as np
import kernels 
import utility_functions as uf
from iteration import Iteration

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
            noise = 0.001,
            utility = 'expected_improvement',   
            verbosity = 1       
        ):


        # Load the kernel function, ensuring user  defined functions pass a 
        # basic set of tests.
        if callable(kernel):
            assert False, "ERROR: kernel function support not implemented yet!"
        else:
            self.kernel  = kernels.RBF() 

        self.noise_kernel = kernels.Noise(noise)
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
                        X,
                        f_X
                    )

            # Evaluate the width at each point in X_test
            # of the postier.
            width = np.atleast_2d(np.sqrt(np.diag(covariance))).T

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
                        best_u,
                        X_test
                    ))

            X   = np.vstack((X, next_x) )

            self.log( "Moving to position: "+ str(next_x))
            f_X = np.vstack((f_X, f(next_x)))
            self.log( "Next position has f: "+ str(f_X[-1]))

        self.log(" Complete." )
        return iterations 



    def is_converged(self, convergence_criteria, param):
        """ 

        """
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

        k_xX = self.kernel(x,X)+self.noise_kernel(x,X)
        k_XX = self.kernel(X,X)+self.noise_kernel(X)
        k_XX_inv = np.linalg.inv(k_XX)
        k_xx = self.kernel(x,x)+self.noise_kernel(x)
        k_Xx = self.kernel(X,x)+self.noise_kernel(X,x)

        mean = np.matmul(k_XX_inv,f_X )
        mean = np.matmul( k_xX, mean)

        sigma = np.matmul(k_XX_inv, k_Xx)
        sigma = np.matmul(k_xX, sigma)
        sigma = k_xx - sigma

        return mean, sigma