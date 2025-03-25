import torch
from torchdiffeq import odeint
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import copy
import numpy as np

def node(t, y, ann, maxi, device=torch.device("cpu")):
    y = y.to(device)
    ann = ann.to(device)

    # Soft clamp the states to be at least EPS
    EPS = 1e-6
    y_clamped = torch.clamp(y, min=EPS)

    y_red = y_clamped / y_clamped.mean()
    mu_pred = ann(y_red).flatten()
    hdm_system = mu_pred * maxi.flatten()**0.5

    return hdm_system

def integrate(growth_NN, y0, tlims, maxi, device = torch.device("cpu")):

    t_new = torch.linspace(tlims[0].item(), tlims[1].item(), 100, dtype=torch.float32, device=device)
    sol = odeint(lambda t, y: node(t, y, growth_NN, maxi), y0, t_new, method = 'implicit_adams', options={'dtype': torch.float32})

    return t_new, sol

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.net = nn.Sequential(

            #Input layer
            nn.Linear(5, 50),
            nn.Tanh(),

            #Hidden layers
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),

            #Output layer
            nn.Linear(50, 5)
        )

    def forward(self, y):
        return self.net(y)

class NODEAM():
    def __init__(self, data, weights):

        #Load the data
        self.data = data

        #Load the NN
        self.weights = weights
        self.node = ANN()
        self.node.load_state_dict(weights)

        #check if self.weights is not an empty array
        # if self.weights is not None:
        #     self.node.load_state_dict(weights)
    
        self.fiterror = None

        #Initialize the aspects needed with data
        self.y0 = torch.tensor(self.data[:, 0], dtype=torch.float32)
        self.tlims = torch.tensor([0, 24*self.data.shape[1]], dtype=torch.float32)
        self.sigma = torch.var(torch.tensor(self.data, dtype=torch.float32), dim=1, keepdim=True)
        self.times = torch.linspace(self.tlims[0], self.tlims[1], self.data.shape[1])

        #Create storage for the augmented dataset
        self.augmented_data = None
        self.augtime = np.linspace(self.tlims[0], self.tlims[1], 20)
    
    def solve(self):

        t, sol = integrate(self.node, self.y0, self.tlims, self.sigma)

        return t, sol.T.detach().numpy()
    
    def solve_ivp(self, y0):

        y0 = torch.tensor(y0, dtype=torch.float32)
        
        t, sol = integrate(self.node, y0, self.tlims, self.sigma)
        
        return t, sol.T.detach().numpy()

    def visual(self):

        t, sol = integrate(self.node, self.y0, self.tlims, self.sigma)

        plt.figure(figsize=(10, 6))
        plt.subplot(1,2,1)
        for i in [0,1,2]:
            plt.plot(t, sol.T[i].detach().numpy())
            plt.scatter(self.times, self.data[i])

        plt.subplot(1,2,2)
        for i in [3,4]:
            plt.plot(t, sol.T[i].detach().numpy())
            plt.scatter(self.times, self.data[i])
        
        plt.show()

    def optimize_params(self, epochs = 10000, return_loss = False):
        'Parameterizes the NN of the NODE model'

        def objective(neuralNet):

            regularization = 0.01

            try:
                t, sol = integrate(neuralNet, self.y0, self.tlims, self.sigma)
            except (ValueError, OverflowError, RuntimeError, TypeError, AssertionError) as e:
                print('Failed to Integrate')
                print(f'Error: {e}')
                return 1e5 + sum(torch.sum(torch.abs(param)) for param in neuralNet.parameters())
            
            #Find the timepoints in the solution that match the data time points
            indices = torch.searchsorted(t,self.times, right=False)
            indices = indices.clamp(max=t.size(0) - 1)

            pred_data = sol[indices].T
            residuals = (torch.from_numpy(self.data) - pred_data)**2 / self.sigma
            loss = residuals.sum() + residuals[0,-1] + residuals[2,-1]

            return loss + regularization * sum(torch.sum(torch.abs(param)) for param in neuralNet.parameters())
        
        if return_loss:
            return objective(self.node)
        
        else:
            print('Starting Loss', objective(self.node))

            #Training loop
            ann = ANN()
            if self.weights is not None:
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
            
            self.node = ANN()  # Reinitialize the ANN object
            self.node.load_state_dict(best_weights)  # Load the best weights
            self.weights = best_weights  # Update the stored weights
            self.fiterror = best_loss  # Store the best loss

            # Ensure gradients are not tracked when computing the final loss
            with torch.no_grad():
                final_loss = objective(self.node)

            # Print results
            print('Best Loss:', best_loss)
            print('Final Loss with Restored Model:', final_loss)

    def augment(self, num_samples, num_timepoints, error):
        'Augments data by solving the ODE system for num_samples samples and num_timepoints timepoints, returns augmented data'

        #Initialize augmented data storage
        aug_data = torch.zeros((num_samples, 5, num_timepoints), dtype=torch.float32)

        #Set first index as optimal sample
        t, sol = integrate(self.node, self.y0, self.tlims, self.sigma)
        indices = torch.linspace(0, len(t)-1, num_timepoints, dtype=torch.int)
        aug_data[0] = sol.T[:, indices]

        for i in range(1, num_samples):
            
            while True:

                #Add noise to initial conditions
                y0 = self.y0 + torch.normal(torch.zeros_like(self.y0), self.y0*error)

                noise_variance = torch.normal(torch.zeros_like(self.sigma), self.sigma*error)

                #Solve ODE system
                try:
                    t, sol = integrate(self.node, y0, self.tlims, self.sigma + noise_variance)
                    break
                except (ValueError, OverflowError, RuntimeError, TypeError, AssertionError) as e:
                    # print('Failed to Integrate')
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