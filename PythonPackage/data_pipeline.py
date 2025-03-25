import numpy as np
import torch
from pathlib import Path

from PythonPackage.models.fpAM import FPAM
from PythonPackage.models.hdAM import HDAM
from PythonPackage.models.nodeAM import NODEAM
from PythonPackage.regression import MPLSR

def collect_augs(h_models, i_models):

    #Create storage tensor
    shape = np.shape(h_models[0].augmented_data)
    tensor = np.zeros((2, 4,*shape))

    #Populate
    for i in range(4):
        tensor[0, i] = h_models[i].augmented_data
        tensor[1, i] = i_models[i].augmented_data
    
    return tensor

class Pipeline():
    def __init__(self, rootpath, name=''):

        self.rootpath = Path(rootpath)

        #Load the experimental data
        self.Hdata = np.load(self.rootpath / 'ExpData' / 'Hdata.npy')
        self.Idata = np.load(self.rootpath / 'ExpData' / 'Idata.npy')

        'Parameter Weights for the Models'
        #Load the fpAM weights
        self.fpmHweights = np.load(self.rootpath / 'parameters' / f'{name}fpmH.npy')
        self.fpmIweights = np.load(self.rootpath / 'parameters' / f'{name}fpmI.npy')

        #Load the hdAM weights
        self.hdmHweights = torch.load(self.rootpath / 'parameters' / f'{name}hdmH.pt')
        self.hdmIweights = torch.load(self.rootpath / 'parameters' / f'{name}hdmI.pt')

        #Load the nodeAM weights
        self.nodeHweights = torch.load(self.rootpath / 'parameters' / f'{name}nodeH.pt')
        self.nodeIweights = torch.load(self.rootpath / 'parameters' / f'{name}nodeI.pt')

        'Augmented Data Sets'
        fpm_aug_data = np.load(self.rootpath / 'AugmentedData' / 'fpam_augmented.npy')
        hdm_aug_data = np.load(self.rootpath / 'AugmentedData' / 'hdam_augmented.npy')
        node_aug_data = np.load(self.rootpath / 'AugmentedData' / 'nodeam_augmented.npy')

        'AM Construction'
        #Construct the AMs
        self.hcell_FPAMs = self.construct_AMs(self.Hdata, self.fpmHweights, fpm_aug_data[0], FPAM)
        self.icell_FPAMs = self.construct_AMs(self.Idata, self.fpmIweights, fpm_aug_data[1], FPAM)
        self.hcell_HDAMs = self.construct_AMs(self.Hdata, self.hdmHweights, hdm_aug_data[0], HDAM)
        self.icell_HDAMs = self.construct_AMs(self.Idata, self.hdmIweights, hdm_aug_data[1], HDAM)
        self.hcell_NODEAMs = self.construct_AMs(self.Hdata, self.nodeHweights, node_aug_data[0], NODEAM)
        self.icell_NODEAMs = self.construct_AMs(self.Idata, self.nodeIweights, node_aug_data[1], NODEAM)

        #Get the importance tensors [AM, cell, feed, metab, times]
        self.importance = None

    def construct_AMs(self, data, weights, augData, modelClass):
        models = []

        for i in range(4):
            if i == 3:
                model = modelClass(data[i,:,:], weights[i])
            else:
                model = modelClass(data[i,:,1:], weights[i])

            model.augmented_data = augData[i]

            models.append(model)

        return models
    
    def training(self, 
                 n_epochs = 1000,
                 method = 'Nelder-Mead',
                 Hcells = True, 
                 Icells = True, 
                 FPAM = False, 
                 HDAM = False, 
                 NODEAM = False, 
                 visualize = False,
                 data_sets = [0,1,2,3]):
        
        if Hcells:
            if FPAM:
                for i in data_sets:
                    self.hcell_FPAMs[i].optimize_params(optimization_method=method)
                    if visualize:
                        self.hcell_FPAMs[i].visual()
                for i in range(4):
                    self.fpmHweights[i] = self.hcell_FPAMs[i].weights

            if HDAM:
                for i in data_sets:
                    self.hcell_HDAMs[i].optimize_params(epochs = n_epochs)
                    if visualize:
                        self.hcell_HDAMs[i].visual()
                for i in range(4):
                    self.hdmHweights[i] = self.hcell_HDAMs[i].weights

            if NODEAM:
                for i in data_sets:
                    self.hcell_NODEAMs[i].optimize_params(epochs = n_epochs)
                    if visualize:
                        self.hcell_NODEAMs[i].visual()
                for i in range(4):
                    self.nodeHweights[i] = self.hcell_NODEAMs[i].weights

        if Icells:
            if FPAM:
                for i in data_sets:
                    self.icell_FPAMs[i].optimize_params(optimization_method=method)
                    if visualize:
                        self.icell_FPAMs[i].visual()
                for i in range(4):
                    self.fpmIweights[i] = self.icell_FPAMs[i].weights

            if HDAM:
                for i in data_sets:
                    self.icell_HDAMs[i].optimize_params(epochs = n_epochs)
                    if visualize:
                        self.icell_HDAMs[i].visual()
                for i in range(4):
                    self.hdmIweights[i] = self.icell_HDAMs[i].weights

            if NODEAM:
                for i in data_sets:
                    self.icell_NODEAMs[i].optimize_params(epochs = n_epochs)
                    if visualize:
                        self.icell_NODEAMs[i].visual()
                for i in range(4):
                    self.nodeIweights[i] = self.icell_NODEAMs[i].weights

    def get_loss_values(self):

        loss_tensor = np.zeros((3,2,4))

        for i in range(4):
            loss_tensor[0,0,i] = self.hcell_FPAMs[i].optimize_params(return_loss=True)
            loss_tensor[0,1,i] = self.icell_FPAMs[i].optimize_params(return_loss=True)

            loss_tensor[1,0,i] = self.hcell_HDAMs[i].optimize_params(return_loss=True)
            loss_tensor[1,1,i] = self.icell_HDAMs[i].optimize_params(return_loss=True)

            loss_tensor[2,0,i] = self.hcell_NODEAMs[i].optimize_params(return_loss=True)
            loss_tensor[2,1,i] = self.icell_NODEAMs[i].optimize_params(return_loss=True)
        
        return loss_tensor

    def augmentation(self, n_aug = 500, num_timepoints = 20, error = [0.1, 0.01, 0.025], Hcells = True, Icells = True, FPAM = True, HDAM = True, NODEAM = True, visualize = False):
        
        if Hcells:
            if FPAM:
                for i in range(4):
                    self.hcell_FPAMs[i].augment(n_aug, num_timepoints, error[0])
                    if visualize:
                        print('Hcell FPAM ', i)
                        self.hcell_FPAMs[i].plot()
            if HDAM:
                for i in range(4):
                    self.hcell_HDAMs[i].augment(n_aug, num_timepoints, error[1])
                    if visualize:
                        print('Hcell HDAM ', i)
                        self.hcell_HDAMs[i].plot()
            if NODEAM:
                for i in range(4):
                    self.hcell_NODEAMs[i].augment(n_aug, num_timepoints, error[2])
                    if visualize:
                        print('Hcell NODEAM ', i)
                        self.hcell_NODEAMs[i].plot()
        if Icells:
            if FPAM:
                for i in range(4):
                    self.icell_FPAMs[i].augment(n_aug, num_timepoints, error[0])
                    if visualize:
                        print('Icell FPAM ', i)
                        self.icell_FPAMs[i].plot()
            if HDAM:
                for i in range(4):
                    self.icell_HDAMs[i].augment(n_aug, num_timepoints, error[1])
                    if visualize:
                        print('Icell HDAM ', i)
                        self.icell_HDAMs[i].plot()
            if NODEAM:
                for i in range(4):
                    self.icell_NODEAMs[i].augment(n_aug, num_timepoints, error[2])
                    if visualize:
                        print('Icell NODEAM ', i)
                        self.icell_NODEAMs[i].plot()

    def importance_regression(self, num_timepoints = 20):

        def get_importance(models, latent_variables = 4, num_timepoints = 20):

            importance_tensor_model = np.zeros((4, 2, num_timepoints))

            for i, model in enumerate(models):
                regression = MPLSR(model, latent_variables)
                regression.get_importance()
                importance_tensor_model[i] = regression.importance

            return importance_tensor_model

        self.importance = np.zeros((
            3, #Models
            2, #Cell Types
            4, #Feed Conditions
            2, #Metabolites
            num_timepoints
            ))

        #FPAM
        self.importance[0, 0] = get_importance(self.hcell_FPAMs)
        self.importance[0, 1] = get_importance(self.icell_FPAMs)
        print('FPAM done')

        #HDAM
        self.importance[1, 0] = get_importance(self.hcell_HDAMs)
        self.importance[1, 1] = get_importance(self.icell_HDAMs)
        print('HDAM done')

        #NODEAM
        self.importance[2, 0] = get_importance(self.hcell_NODEAMs)
        self.importance[2, 1] = get_importance(self.icell_NODEAMs)
        print('NODEAM done')

        print('Importance done \n')

    def mse_regression(self, FPAM = True, HDAM = True, NODEAM = True):
        
        mse_tensor = np.zeros((3,2,2,4,2,2)) #model, cell, feed, cqa

        #Helper function
        def mse(models):
            
            mse_tensor_models = np.zeros((4, 2, 2))

            for i, model in enumerate(models):
                regression = MPLSR(model, 4)
                mse = regression.mse_testing()
                mse_tensor_models[i] = mse

            return mse_tensor_models
        
        #FPAM  
        if FPAM:
            mse_tensor[0, 0] = mse(self.hcell_FPAMs)
            print('change to i cell')
            mse_tensor[0, 1] = mse(self.icell_FPAMs)
            print('FPAM done')

        #HDAM
        if HDAM:
            mse_tensor[1, 0] = mse(self.hcell_HDAMs)
            print('change to i cell')
            mse_tensor[1, 1] = mse(self.icell_HDAMs)
            print('HDAM done')

        #NODEAM
        if NODEAM:
            mse_tensor[2, 0] = mse(self.hcell_NODEAMs)
            print('change to i cell')
            mse_tensor[2, 1] = mse(self.icell_NODEAMs)
            print('NODEAM done')

        print('MSE done \n')

        return mse_tensor
    
    def validation_predict(self, X_hcell, X_icell, ics):

        #Generate a reference data set with the ICs and get the predictor
        def prepare_validation(model, Xtest, hdam = False, fpam = False):

            predictor = MPLSR(model)

            y0 = None

            if hdam:
                #reorder to 0, 1, 3, 4, 2
                y0 = np.array([ics[0], ics[1], ics[3], 2.6, ics[2]])
            elif fpam:
                y0 = np.array([ics[0], ics[1], ics[2], ics[3], ics[4], 100, 0.01])
            else:
                y0 = np.array([ics[0], ics[1], ics[2], ics[3], ics[4]])
            
            t, y = model.solve_ivp(y0)
            
            x_sparse = np.zeros((5, 20)) #5 metabolites, 20 timepoints

            #get 20 evenly spaced timepoints from the y array
            x_sparse = y[:,::int(y.shape[1]/20)]

            if hdam:
                Xtrain = x_sparse[[2,3]]
            else:
                Xtrain = x_sparse[[3,4]]

            return predictor.predict(Xtrain, Xtest)

        cqa_predictions = np.zeros((3, 2, 2, 20)) # 3 models, 2 cell types, 2 CQAs, 20 timepoints

        print('fpam')
        cqa_predictions[0, 0] = prepare_validation(self.hcell_FPAMs[3], X_hcell, fpam = True)
        cqa_predictions[0, 1] = prepare_validation(self.icell_FPAMs[3], X_icell, fpam = True)

        print('hdam')
        cqa_predictions[1, 0] = prepare_validation(self.hcell_HDAMs[2], X_hcell, hdam = True)
        cqa_predictions[1, 1] = prepare_validation(self.icell_HDAMs[2], X_icell, hdam = True)

        print('nodeam')
        cqa_predictions[2, 0] = prepare_validation(self.hcell_NODEAMs[3], X_hcell)
        cqa_predictions[2, 1] = prepare_validation(self.icell_NODEAMs[3], X_icell)

        return cqa_predictions
    
    def validation_predict2(self, X_hcell, X_icell):

        #Hcells
        H_fpAM_predictor = MPLSR(self.hcell_FPAMs[3])
        H_hdAM_predictor = MPLSR(self.hcell_HDAMs[3])
        H_nodeAM_predictor = MPLSR(self.hcell_NODEAMs[3])

        #Icells
        I_fpAM_predictor = MPLSR(self.icell_FPAMs[3])
        I_hdAM_predictor = MPLSR(self.icell_HDAMs[3])
        I_nodeAM_predictor = MPLSR(self.icell_NODEAMs[3])

        cqa_predictions = np.zeros((3, 2, 2, 20)) # 3 models, 2 cell types, 2 CQAs, 20 timepoints

        #Predictions
        print('fpam')
        cqa_predictions[0, 0] = H_fpAM_predictor.predict(X_hcell, self.hcell_FPAMs[3].augmented_data[0])
        cqa_predictions[0, 1] = I_fpAM_predictor.predict(X_icell, self.icell_FPAMs[3].augmented_data[0])

        print('hdam')
        cqa_predictions[1, 0] = H_hdAM_predictor.predict(X_hcell, self.hcell_HDAMs[3].augmented_data[0])
        cqa_predictions[1, 1] = I_hdAM_predictor.predict(X_icell, self.icell_HDAMs[3].augmented_data[0])

        print('nodeam')
        cqa_predictions[2, 0] = H_nodeAM_predictor.predict(X_hcell, self.hcell_NODEAMs[3].augmented_data[0])
        cqa_predictions[2, 1] = I_nodeAM_predictor.predict(X_icell, self.icell_NODEAMs[3].augmented_data[0])

        return cqa_predictions

    def visualizeall(self, Hcells = True, Icells = True, FPAM = False, HDAM = False, NODEAM = False):

        if Hcells:
            if FPAM:
                for i in range(4):
                    print('FPAM Hcell ', i)
                    self.hcell_FPAMs[i].visual()
            if HDAM:
                for i in range(4):
                    print('HDAM Hcell ', i)
                    self.hcell_HDAMs[i].visual()
            if NODEAM:
                for i in range(4):
                    print('NODEAM Hcell ', i)
                    self.hcell_NODEAMs[i].visual()
        if Icells:
            if FPAM:
                for i in range(4):
                    print('FPAM Icell ', i)
                    self.icell_FPAMs[i].visual()
            if HDAM:
                for i in range(4):
                    print('HDAM Icell ', i)
                    self.icell_HDAMs[i].visual()
            if NODEAM:
                for i in range(4):
                    print('NODEAM Icell ', i)
                    self.icell_NODEAMs[i].visual()

    def visualize_augmented_data(self, Hcells = True, Icells = True, FPAM = False, HDAM = False, NODEAM = False):

        if Hcells:
            if FPAM:
                for i in range(4):
                    print('FPAM Hcell ', i)
                    self.hcell_FPAMs[i].plot()
            if HDAM:
                for i in range(4):
                    print('HDAM Hcell ', i)
                    self.hcell_HDAMs[i].plot()
            if NODEAM:
                for i in range(4):
                    print('NODEAM Hcell ', i)
                    self.hcell_NODEAMs[i].plot()
        if Icells:
            if FPAM:
                for i in range(4):
                    print('FPAM Icell ', i)
                    self.icell_FPAMs[i].plot()
            if HDAM:
                for i in range(4):
                    print('HDAM Icell ', i)
                    self.icell_HDAMs[i].plot()
            if NODEAM:
                for i in range(4):
                    print('NODEAM Icell ', i)
                    self.icell_NODEAMs[i].plot()

    def save_parameters(self, name, FPAM = False, HDAM = False, NODEAM = False):
        if FPAM:
            fpmH_path = self.rootpath / 'parameters' / f'{name}fpmH.npy'
            fpmI_path = self.rootpath / 'parameters' / f'{name}fpmI.npy'
            np.save(fpmH_path, self.fpmHweights)
            np.save(fpmI_path, self.fpmIweights)

        if HDAM:
            hdmH_path = self.rootpath / 'parameters' / f'{name}hdmH.pt'
            hdmI_path = self.rootpath / 'parameters' / f'{name}hdmI.pt'
            torch.save(self.hdmHweights, hdmH_path)
            torch.save(self.hdmIweights, hdmI_path)

        if NODEAM:
            nodeH_path = self.rootpath / 'parameters' / f'{name}nodeH.pt'
            nodeI_path = self.rootpath / 'parameters' / f'{name}nodeI.pt'
            torch.save(self.nodeHweights, nodeH_path)
            torch.save(self.nodeIweights, nodeI_path)

    def save_augmented_datasets(self, name, FPAM = False, HDAM = False, NODEAM = False):

        if FPAM:
            fpam_augs = collect_augs(self.hcell_FPAMs, self.icell_FPAMs)
            path = self.rootpath / 'AugmentedData' / f'{name}fpam_augmented.npy'
            np.save(path, fpam_augs)

        if HDAM:
            hdam_augs = collect_augs(self.hcell_HDAMs, self.icell_HDAMs)
            path = self.rootpath / 'AugmentedData' / f'{name}hdam_augmented.npy'
            np.save(path, hdam_augs)

        if NODEAM:
            nodeam_augs = collect_augs(self.hcell_NODEAMs, self.icell_NODEAMs)
            path = self.rootpath / 'AugmentedData' / f'{name}nodeam_augmented.npy'
            np.save(path, nodeam_augs)
    
    def save_VIP_tensor(self, name):
        path = self.rootpath / 'Importance' / f'{name}VIP_tensor.npy'
        np.save(path, self.importance)