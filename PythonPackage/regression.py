import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from tensorly.regression import CP_PLSR

from PythonPackage.models.fpAM import FPAM
from PythonPackage.models.hdAM import HDAM
from PythonPackage.models.nodeAM import NODEAM

def getVIP(X, Y, nLVs):

    # Fit Tensor PLS model
    tpls = CP_PLSR(n_components=nLVs)
    try:
        tpls.fit(X, Y)
    except ValueError as e:
        print("LinAlgError: Singular matrix encountered. Skipping this iteration.", e)
        return ValueError

    # Extract factor matrices and coefficients
    X_factors = tpls.X_factors  # X mode decompositions
    Y_factors = tpls.Y_factors  # Y mode decompositions
    coef = tpls.coef_  # Regression coefficients

    # Ensure factors exist
    if X_factors is None or Y_factors is None or coef is None:
        raise ValueError("CP_PLSR did not extract factors or coefficients properly.")

    # Step 1: Extract analogous components
    T = X_factors[0]  # Analogous to x_scores_
    W = X_factors[1]  # Mode-1 (2, nLVs) - Variable importance (Glucose & Lactate)
    P = X_factors[2]  # Mode-2 (20, nLVs) - Timepoint importance

    # Step 2: Compute explained variance per latent variable
    T_ss = np.sum(T ** 2, axis=0)  # Sum of squares of scores
    P_ss = np.sum(P ** 2, axis=0)  # Sum of squares of loadings
    total_variance = np.sum(T ** 2)
    explained_variance = (T_ss * P_ss) / total_variance

    # Step 3: Compute feature importance properly
    W_squared = W**2  # (2, nLVs)
    P_squared = P**2  # (20, nLVs)
    
    # Distribute feature importance over Glucose/Lactate and 20 timepoints
    feature_importance = np.dot(W_squared, P_squared.T)  # (2, 20)

    # Flatten the (2, 20) structure into a 1D array of shape (40,)
    feature_importance = feature_importance.flatten()

    # Step 4: Compute VIP scores per feature
    n_features = X.shape[1] * X.shape[2]  # 2 × 20 = 40 features
    vip = np.sqrt(n_features * feature_importance / np.sum(feature_importance))

    return vip

def makeBoolean(vip):
    vip[vip < 1] = 0
    vip[vip > 1] = 1
    return vip

from sklearn.preprocessing import StandardScaler

class MPLSR():
    def __init__(self, augmentationModel, latent_variables=4):
        
        self.LVs = latent_variables
        self.pls_VCD = PLSRegression(n_components=self.LVs)
        self.pls_Pluri = PLSRegression(n_components=self.LVs)

        self.scaler_Y1 = StandardScaler()  # Scaler for Y[:,1] (pluripotency)

        # Data for the augmentation model
        if type(augmentationModel) in [FPAM, NODEAM]:
            self.X = np.zeros((augmentationModel.augmented_data.shape[0], 2, 20))
            self.X[:,0] = augmentationModel.augmented_data[:,3,:]  # Glucose
            self.X[:,1] = augmentationModel.augmented_data[:,4,:]  # Lactate

            self.Y = np.zeros((augmentationModel.augmented_data.shape[0], 2))
            final_Xv = augmentationModel.augmented_data[:,0,-1]
            final_Xv[final_Xv < 0] = 0
            self.Y[:,0] = final_Xv  # Viable cell density

            final_Xnan = augmentationModel.augmented_data[:,2,-1]
            final_Xnan[final_Xnan < 0] = 0
            for i in range(len(final_Xv)):
                if final_Xv[i] == 0 or final_Xnan[i] > final_Xv[i]:
                    self.Y[:,0][i] = 0
                else:
                    self.Y[:,1][i] = final_Xnan[i] / final_Xv[i]  # Pluripotency fraction

        elif type(augmentationModel) in [HDAM]:
            self.X = np.zeros((augmentationModel.augmented_data.shape[0], 2, 20))
            self.X[:,0] = augmentationModel.augmented_data[:,2,:]  # Glucose
            self.X[:,1] = augmentationModel.augmented_data[:,3,:]  # Lactate

            self.Y = np.zeros((augmentationModel.augmented_data.shape[0], 2))
            final_Xv = augmentationModel.augmented_data[:,0,-1]
            final_Xv[final_Xv < 0] = 0
            self.Y[:,0] = final_Xv  # Viable cell density

            final_Xnan = augmentationModel.augmented_data[:,4,-1]
            final_Xnan[final_Xnan < 0] = 0
            for i in range(len(final_Xv)):
                if final_Xv[i] == 0 or final_Xnan[i] > final_Xv[i]:
                    self.Y[:,0][i] = 0
                else:
                    self.Y[:,1][i] = final_Xnan[i] / final_Xv[i]  # Pluripotency fraction

        # Reshape X
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], self.X.shape[2])

        self.importance = np.zeros((2,20))

    def get_importance(self, n_montecarlo = 500, subset_size = 0.5):
        
        vip_booleans = np.zeros((n_montecarlo, 40))

        for i in range(n_montecarlo):

            # Randomly sample 0.5 of the data points
            indices = np.random.choice(self.X.shape[0], int(subset_size * self.X.shape[0]), replace=False)
            Xsubset = self.X[indices]
            Ysubset = self.Y[indices]

            # Compute VIP scores for Tensor PLSR
            vip_tensor = getVIP(Xsubset, Ysubset, nLVs=4)
            vip_tensor = makeBoolean(vip_tensor)

            # Store VIP scores in 3D array
            vip_booleans[i] = vip_tensor

        # Average VIP scores across all iterations
        avg_vip = np.mean(vip_booleans, axis=0)

        # Reshape to 2 x 20
        self.importance = avg_vip.reshape(2, 20)

    def mse_testing(self):

        mse = np.zeros((2))

        #Set up the model
        pls = CP_PLSR(n_components=self.LVs)

        #Divide the data into training and test, then train the model
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size = 0.2, shuffle = True)
        pls.fit(X_train, y_train)

        #Predict with the model and compare to the validation data
        y_pred = pls.predict(X_test)

        mape1 = mean_absolute_percentage_error(y_test[:,0], y_pred[:,0])
        mse1 = mean_squared_error(y_test[:,0], y_pred[:,0])

        mape2 = mean_absolute_percentage_error(y_test[:,1], y_pred[:,1])
        mse2 = mean_squared_error(y_test[:,1], y_pred[:,1])

        output = np.array([[mape1, mape2],[mse1, mse2]])
        
        return output