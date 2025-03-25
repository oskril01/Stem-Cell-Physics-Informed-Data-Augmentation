import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def ode_system(t, y, constants):
    #Variables
    Xv = y[0]
    Xt = y[1]
    NAN = y[2]
    GLC = y[3]
    LAC = y[4]
    GLN = y[5]
    AMM = y[6]

    # Perturb constants [3, 4, 5, 7, 10, 11, 12]
    Kglc = constants[0]
    Kgln = constants[1]
    Kdgln = constants[2]
    KIlac = constants[3] # Perturb
    mumax = constants[4] # Perturb
    Yxglc = constants[5] # Perturb
    Yxgln = constants[6]
    mglc = constants[7] # Perturb
    alpha1 = constants[8]
    alpha2 = constants[9]
    Ylacglc = constants[10] # Perturb
    kdiff = constants[11] # Perturb
    mudmax = constants[12] # Perturb
    Kdamm = constants[13]
    KIamm = constants[14]
    Yammgln = constants[15]
    
    n = 2

    #Define Functions
    flim = (GLC/(Kglc+GLC)) * (GLN/(Kgln+GLN))
    finh = (KIlac/(KIlac+LAC)) * (KIamm/(KIamm+AMM))
    mu = mumax * flim * finh
    mud = mudmax / (1+np.power((Kdamm/AMM),n))
    mgln = (alpha1*GLN) / (alpha2 + GLN)
    Qglc = (mu/Yxglc) + mglc
    Qgln = (mu/Yxgln) + mgln
    Qlac = Ylacglc*Qglc
    Qamm = Yammgln*Qgln
    
    #Viable Cells
    dXvdt = (mu - mud)*Xv
    
    #Total Cells
    dXtdt = mu*Xv
    
    #Glucose
    if GLC > 0:
        dGLCdt = -Qglc*Xv
    else:
        dGLCdt = 0
    
    #Glutamine
    if GLN > 0:
        dGLNdt = -Qgln*Xv - Kdgln*GLN #10
    else:
        dGLNdt = 0

    #Ammonia
    if AMM > 0:
        dAMMdt = Qamm*Xv + Kdgln*GLN
    else:
        dAMMdt = 0
    
    #Lactate
    if LAC > 0:
        dLACdt = Qlac*Xv
    else:
        dLACdt = 0
    
    #NANOG
    dNANdt = (mu - mud - kdiff)*Xv
    
    #Form System
    ode_system = np.array([dXvdt, dXtdt, dNANdt, dGLCdt, dLACdt, dGLNdt, dAMMdt])
    return ode_system

def integrate(constants, y0, tlims):
    sol = solve_ivp(ode_system, tlims, y0, method='Radau', t_eval=np.linspace(tlims[0], tlims[1], 100), args=(constants,))
    return sol

class FPAM():
    def __init__(self, data, weights):
        self.data = data
        self.weights = weights
        self.augmented_data = None

        self.y0 = np.concatenate((self.data[:, 0], np.array([100, 0.01])))
        self.tlims = (0, 24*self.data.shape[1])
        self.sigma = np.var(self.data, axis=1, keepdims=True)
        self.times = np.linspace(self.tlims[0], self.tlims[1], self.data.shape[1])
        self.augtime = np.linspace(self.tlims[0], self.tlims[1], 20)
        self.fiterror = None

    def solve(self):

        sol = integrate(self.weights, self.y0, self.tlims)

        return sol.t, sol.y
    
    def solve_ivp(self, y0):

        sol = integrate(self.weights, y0, self.tlims)
        
        return sol.t, sol.y

    def visual(self):

        sol = integrate(self.weights, self.y0, self.tlims)
        #print the info from integrate

        plt.figure(figsize=(10, 6))
        plt.subplot(1,2,1)
        for i in [0,1,2]:
            plt.plot(sol.t, sol.y[i])
            plt.scatter(self.times, self.data[i])

        plt.subplot(1,2,2)
        for i in [3,4]:
            plt.plot(sol.t, sol.y[i])
            plt.scatter(self.times, self.data[i])
        
        plt.show()

    def optimize_params(self, optimization_method = 'Nelder-Mead', return_loss = False):
        'Optimizes weights using the data'

        #Max Xt index to ignore after Xt decreases
        max_xt = np.argmax(self.data[1]) + 1
        mask = np.ones_like(self.data, dtype=bool)
        mask[1, max_xt:] = False
        
        #Define loss function for optimization
        def loss(constants):

            sol = integrate(np.abs(constants), self.y0, self.tlims)

            if sol.success == True:
                
                #Find the timepoints in the solution that match the data time points
                differences = np.abs(sol.t.reshape(-1, 1) - self.times)
                closest_indices = np.argmin(differences, axis=0)

                pred_data = sol.y[:, closest_indices]

                residuals = (self.data - pred_data[:5]) ** 2 / self.sigma

                # Weight the final data points higher
                loss = np.sum(residuals) + (residuals[1,-1] + residuals[2, -1])

                return loss

            else:
                print('Integration Failed')
                return 1e25

        if return_loss:
            return loss(self.weights)
        
        else:
            print('Starting Loss', loss(self.weights))
            optimization_solution = minimize(loss, self.weights, method = optimization_method)
            self.weights = np.abs(optimization_solution.x)
            print('Ending Loss', loss(self.weights))
            self.fiterror = loss(self.weights)

    def augment(self, num_samples, num_timepoints, error):
        'Augments data by solving the ODE system for num_samples samples and num_timepoints timepoints, returns augmented data'

        #Initialize augmented data storage
        aug_data = np.zeros((num_samples, 5, num_timepoints))

        #Set first index as optimal sample
        solution = integrate(self.weights, self.y0, self.tlims)
        indices = np.linspace(0, len(solution.t)-1, num_timepoints, dtype=int)
        aug_data[0] = solution.y[:5, indices]

        for i in range(1, num_samples):
            
            #Add noise to initial conditions
            y0 = self.y0 + np.random.normal(np.zeros_like(self.y0), self.y0*error, size=self.y0.shape)

            #Sample constants from normal distribution around weights
            # ONLY Perturb constants with the following indices [3, 4, 5, 7, 10, 11, 12]
            constants = self.weights.copy()
            sample = np.random.normal(self.weights, np.abs(self.weights*error), size=self.weights.shape)
            constants[[3, 4, 5, 7, 10, 11, 12]] = sample[[3, 4, 5, 7, 10, 11, 12]]

            #Solve ODE system
            solution = integrate(constants, y0, self.tlims).y[:5]

            while solution.shape[1] < 100:
                #Add noise to initial conditions
                y0 = self.y0 + np.random.normal(np.zeros_like(self.y0), self.y0*error, size=self.y0.shape)

                #Sample constants from normal distribution around weights
                constants = self.weights.copy()
                sample = np.random.normal(self.weights, np.abs(self.weights*error), size=self.weights.shape)
                constants[[3, 4, 5, 7, 10, 11, 12]] = sample[[3, 4, 5, 7, 10, 11, 12]]

                #Solve ODE system
                solution = integrate(constants, y0, self.tlims).y[:5]

            aug_data[i] = solution[:, indices]

        self.augmented_data = aug_data
        self.augtime = np.linspace(self.tlims[0], self.tlims[1], num_timepoints)

    def plot(self):
        'Plots the data'

        if self.augmented_data is None:
            print('No augmented data to plot')
            return

        #Plot the optimal solution and the data points
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1,2,1)
        for i in [0,1,2]:
            plt.scatter(self.times, self.data[i])
            plt.plot(self.augtime, self.augmented_data[0,i])
            for j in range(1, self.augmented_data.shape[0]):
                plt.plot(self.augtime, self.augmented_data[j,i], 'k', alpha=0.005)

        plt.subplot(1,2,2)
        for i in [3,4]:
            plt.scatter(self.times, self.data[i])
            plt.plot(self.augtime, self.augmented_data[0,i])
            for j in range(1, self.augmented_data.shape[0]):
                plt.plot(self.augtime, self.augmented_data[j,i], 'k', alpha=0.005)

        plt.show()