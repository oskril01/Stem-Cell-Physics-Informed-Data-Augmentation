import torch
from torchdiffeq import odeint
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import copy
import numpy as np

mu_max = torch.tensor([
    7.4e-3,#Xv
    7.4e-3,#Xt
    9e-8,#GLC
    9e-8,#LAC
    7.4e-3#NAN
], dtype=torch.float32)

def hd_model(t, y, ann, mu_max=mu_max, device = torch.device("cpu")):

    # Use the following order of y-values:
    # y[0] = Xv
    # y[1] = Xt
    # y[2] = GLC
    # y[3] = LAC
    # y[4] = NAN
    
    #put all inputs to the device 
    y = y.to(device)
    mu_max = mu_max.to(device)
    ann = ann.to(device)

    y_red = y/y.mean()

    # Predict mu using the current state of y
    mu_pred = ann(y_red).flatten()

    # Stoichiometry matrix initialization
    stoich = torch.tensor([1, 1, -1, 1, 1], dtype=torch.float32, device=device)

    # Combine to create the system of ODEs
    hdm_system = mu_max * mu_pred * stoich * y[0]

    #Constraints
    for i in range(5):
        if y[i] <= 0:
            hdm_system[i] = 0

    return hdm_system

def integrate(growth_NN, y0, tlims, mu = mu_max, device = torch.device("cpu")):

    t_new = torch.linspace(tlims[0].item(), tlims[1].item(), 100, dtype=torch.float32, device=device)
    sol = odeint(lambda t, y: hd_model(t, y, growth_NN, mu), y0, t_new)

    return t_new, sol

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 10),
            nn.Tanh(),
            nn.Linear(10, 5)
        )

    def forward(self, y):
        return self.net(y)

class HDAM():
    def __init__(self, data, weights):
        
        #Initialize the data but swap 
        self.data = data[[0,1,3,4,2],:]
        self.weights = weights
        self.fiterror = None

        #Load the ANN
        self.growth_NN = ANN()
        self.growth_NN.load_state_dict(self.weights)

        #Initialize the aspects needed with data
        self.y0 = torch.tensor(self.data[:, 0], dtype=torch.float32)
        self.tlims = torch.tensor([0, 24*self.data.shape[1]], dtype=torch.float32)
        self.sigma = torch.var(torch.tensor(self.data, dtype=torch.float32), dim=1, keepdim=True)
        self.times = torch.linspace(self.tlims[0], self.tlims[1], self.data.shape[1])

        #Create storage for the augmented dataset
        self.augmented_data = None
        self.augtime = np.linspace(self.tlims[0], self.tlims[1], 20)

    def solve(self):

        t, sol = integrate(self.growth_NN, self.y0, self.tlims)

        return t, sol.T.detach().numpy()
    
    def solve_ivp(self, y0):

        y0 = torch.tensor(y0, dtype=torch.float32)
        
        t, sol = integrate(self.growth_NN, y0, self.tlims)
        
        return t, sol.T.detach().numpy()

    def visual(self):

        t, sol = integrate(self.growth_NN, self.y0, self.tlims)

        plt.figure(figsize=(10, 6))
        plt.subplot(1,2,1)
        for i in [0,1,4]:
            plt.plot(t, sol.T[i].detach().numpy())
            plt.scatter(self.times, self.data[i])

        plt.subplot(1,2,2)
        for i in [2,3]:
            plt.plot(t, sol.T[i].detach().numpy())
            plt.scatter(self.times, self.data[i])
        
        plt.show()

    def optimize_params(self, epochs = 1000, return_loss = False):
        'Optimizes the parameters of the growth NN using the data'

        def objective(neuralNet):

            regularization = 0.05

            try:
                t, sol = integrate(neuralNet, self.y0, self.tlims)
            except (ValueError, OverflowError, RuntimeError, TypeError, AssertionError) as e:
                print('Failed to Integrate')
                print(f'Error: {e}')
                return  1e5 + sum(torch.sum(torch.abs(param)) for param in neuralNet.parameters())
            
            #Find the timepoints in the solution that match the data time points
            indices = torch.searchsorted(t,self.times, right=False)
            indices = indices.clamp(max=t.size(0) - 1)

            pred_data = sol[indices].T
            residuals = (torch.from_numpy(self.data) - pred_data)**2 / self.sigma
            loss = residuals.sum() + residuals[0,-1] + residuals[4,-1] + regularization * sum(torch.sum(torch.abs(param)) for param in neuralNet.parameters())

            return loss
        
        if return_loss:
            return objective(self.growth_NN)

        else:
            print('Starting Loss', objective(self.growth_NN))

            #Training loop
            ann = ANN()
            ann.load_state_dict(self.weights)
            optimizer = Adam(ann.parameters(), lr=0.001)
            loss_values = []

            best_loss = float('inf')
            best_weights = None

            for epoch in range(epochs):
                optimizer.zero_grad()
                loss = objective(ann)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
                print(f'Epoch: {epoch}, Loss: {loss.item()}')

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_weights = copy.deepcopy(ann.state_dict())


            self.growth_NN = ANN()  # Reinitialize the ANN object
            self.growth_NN.load_state_dict(best_weights)  # Load the best weights
            self.weights = best_weights  # Update the stored weights
            self.fiterror = best_loss  # Store the best loss

            print('Ending Loss', objective(self.growth_NN))

    def augment(self, num_samples, num_timepoints, error):
        'Augments data by solving the ODE system for num_samples samples and num_timepoints timepoints, returns augmented data'

        #Initialize augmented data storage
        aug_data = torch.zeros((num_samples, 5, num_timepoints), dtype=torch.float32)

        #Set first index as optimal sample
        t, sol = integrate(self.growth_NN, self.y0, self.tlims)
        indices = torch.linspace(0, len(t)-1, num_timepoints, dtype=torch.int)
        aug_data[0] = sol.T[:, indices]

        for i in range(1, num_samples):
            
            while True:

                #Add noise to initial conditions
                y0 = self.y0 + torch.normal(torch.zeros_like(self.y0), self.y0*error)
                
                #Sample mu_max from normal distribution around mu_max
                mu = torch.normal(mu_max, mu_max*error)

                #Solve ODE system
                try:
                    t, sol = integrate(self.growth_NN, y0, self.tlims, mu)
                    break
                except (ValueError, OverflowError, RuntimeError, TypeError, AssertionError) as e:
                    # print(f'Error: {e}')
                    continue

            aug_data[i] = sol.T[:, indices]

        self.augmented_data = aug_data.detach().numpy()
        self.augtime = torch.linspace(self.tlims[0], self.tlims[1], num_timepoints).detach().numpy()

    def plot(self):
        'Plots the data'

        if self.augmented_data is None:
            print('No augmented data to plot')
            return
        
        #Plot the optimal solution and the data points
        plt.figure(figsize=(10, 6))

        plt.subplot(1,2,1)
        for i in [0,1,4]:
            plt.scatter(self.times, self.data[i])
            plt.plot(self.augtime, self.augmented_data[0,i])
            for j in range(1, self.augmented_data.shape[0]):
                plt.plot(self.augtime, self.augmented_data[j,i], 'k', alpha=0.005)

        plt.subplot(1,2,2)
        for i in [2,3]:
            plt.scatter(self.times, self.data[i])
            plt.plot(self.augtime, self.augmented_data[0,i])
            for j in range(1, self.augmented_data.shape[0]):
                plt.plot(self.augtime, self.augmented_data[j,i], 'k', alpha=0.005)

        plt.show()