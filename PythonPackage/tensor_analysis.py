import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from skfda.inference.anova import oneway_anova
from skfda import FDataGrid

def mode_data(tensor):

    # Get the shapes of all modes except time
    dimensions = tensor.shape[:-1]  
    time_points = tensor.shape[-1]  

    mode_data = []

    # Collect temporal VIP data for each mode
    for mode in range(len(dimensions)):
        other_modes = [dimensions[i] for i in range(len(dimensions)) if i != mode]
        mode_index_count = dimensions[mode]  
        other_index_count = np.prod(other_modes)

        # Storage array: (groups, samples per group, time points)
        mode_vectors = np.zeros((mode_index_count, other_index_count, time_points))

        # Iterate through each index in the target mode
        for idx in range(dimensions[mode]):
            col = 0  # Column counter for other indices

            # Iterate over all other indices
            for other_indices in np.ndindex(*other_modes):
                indices = list(other_indices)
                indices.insert(mode, idx)  # Insert the fixed mode index at the correct place
                mode_vectors[idx, col, :] = tensor[tuple(indices)]  # Extract time series
                col += 1

        # Append reshaped data for FDA-ANOVA
        mode_data.append(mode_vectors)

    return mode_data

def fda_anova(mode_data, lp=2):

    nmodes = len(mode_data)  # Number of modes to analyze
    p_values = []
    stat_values = []

    for i in range(nmodes):
        data = mode_data[i]  # Extract mode-specific data
        n, m, t = data.shape  # (groups, samples per group, time points)

        # Generate time grid for functional data
        time = np.linspace(0, 1, t)

        # Create functional data objects for each group
        fd_groups = [FDataGrid(data[j], grid_points=time) for j in range(n)]

        # Perform FDA-ANOVA
        stat, p = oneway_anova(*fd_groups, p = lp)  # Ensure correct argument unpacking

        p_values.append(p)
        stat_values.append(stat)

    return p_values, stat_values

class VIP_tensor():
    def __init__(self, path):

        #Load the VIP tensor
        self.tensor = np.load(path)
        self.mode_labels = ['Augmentation Model', 'Cell Type', 'Glucose Feed (mM)', 'Glycolytic Indicator']
        self.mode_sublabels = [['fpAM', 'hdAM', 'nodeAM'], ['hESCs', 'iPSCs'], ['1.0', '5.0', '17.5', '20.0'], ['Glucose', 'Lactate']]

        #check for nans in the tensor
        self.nan_indices = np.argwhere(np.isnan(self.tensor))

        #replace nans with zeros
        #self.tensor[self.nan_indices[:,0], self.nan_indices[:,1], self.nan_indices[:,2], self.nan_indices[:,3], self.nan_indices[:,4]] = 0

        #Get the results of the FDA-ANOVA to compare the modes
        self.fpam_tensor = self.tensor[0]
        self.hdam_tensor = self.tensor[1]
        self.nodeam_tensor = self.tensor[2]

        self.p_values = []
        self.f_stats = []

        for i in range(3):
            p, f = fda_anova(mode_data(self.tensor[i, :, :3]))
            
            self.p_values.append(p)
            self.f_stats.append(f)
        
        self.model_pval = fda_anova(mode_data(self.tensor))
        print(self.model_pval[0][0])

    def fda_anova_results(self):

        return self.p_values, self.f_stats

    def average_out(self, avg_indices = (1,)):

        #Average over insignificant indices then collect new mode data
        avg_data = np.mean(self.tensor, axis = avg_indices)
        moded_data = mode_data(avg_data)

        #Collect labels for sig modes
        mode_labels = [self.mode_labels[i] for i in range(4) if i not in avg_indices]
        sublabels = [self.mode_sublabels[i] for i in range(4) if i not in avg_indices]

        avgs = []
        # Process each mode for average trajectories
        for i in range(len(mode_labels)):
            mode = moded_data[i]
            n_conds = mode.shape[0]
            avg_store = np.zeros((n_conds, mode.shape[2]))

            for j in range(n_conds):
                avg_store[j] = np.mean(mode[j], axis=0)

            avgs.append(avg_store)

        # Store the averaged data with labels in a dictionary
        avg_dict = {}
        for i, label in enumerate(mode_labels):
            avg_dict[label] = {sublabels[i][j]: avgs[i][j] for j in range(len(sublabels[i]))}

        return avg_dict

    def plot_mode_heatmaps(self, indices = (1,2,3), colormap= plt.cm.Greys, scale=1, size = (12, 8), colorlabel = 'Importance'):

        data_dict = self.average_out(avg_indices = indices)

        times = np.linspace(0, 144, 20)
        times = np.around(times, 1)
        time_labels = [str(i) for i in times]

        #get the maximum value in the data_dict values
        max_val = 0
        for mode in data_dict.values():
            for inner in mode.values():
                if np.max(inner) > max_val:
                    max_val = np.max(inner)

        # Step 1: Find the global min and max for consistent color scaling
        global_min = 0
        global_max = max_val

        # Step 2: Set font and layout configurations
        sns.set_context("talk", font_scale=scale)  # Sets font scaling
        
        # Calculate relative heights for subplots based on the number of inner entries in each mode
        row_heights = [len(inner_dict) for inner_dict in data_dict.values()]

        # Step 3: Set up the figure and axes with specified row heights
        fig, axs = plt.subplots(len(data_dict), 1, figsize=size, 
                                gridspec_kw={'height_ratios': row_heights}, constrained_layout=True, dpi = 300)
        fig.suptitle("Average Temporal Importance")

        # Ensure axs is iterable, even if there's only one mode
        if len(data_dict) == 1:
            axs = [axs]
        
        # Step 4: Plot each mode as a separate heatmap plot
        for idx, (mode_label, inner_dict) in enumerate(data_dict.items()):
            n_rows = len(inner_dict)  # Number of inner entries for this mode
            n_cols = len(time_labels)  # Should match the number of time points in each inner entry

            # Prepare an array to hold the heatmap data for this mode (shape: n_rows x n_cols)
            heatmap_data = np.array([inner_dict[inner_label] for inner_label in inner_dict])

            # Plot the heatmap with shared vmin and vmax
            cax = axs[idx].imshow(heatmap_data, aspect="auto", cmap=colormap, vmin=global_min, vmax=global_max)
            
            # Set labels and ticks
            axs[idx].set_ylabel(mode_label)
            axs[idx].yaxis.set_label_coords(-0.15, 0.5)
            axs[idx].set_yticks(np.arange(n_rows))
            axs[idx].set_yticklabels(list(inner_dict.keys()), rotation=0)
            
            # Remove x-axis labels if not the bottom plot
            if idx != len(data_dict) - 1:
                axs[idx].set_xticks([])
            else:
                axs[idx].set_xticks(np.arange(0, len(time_labels), 3))  # Every 5 ticks
                axs[idx].set_xticklabels(time_labels[::3], rotation=0)  # Label every 5th tick                ax.set_xticklabels(time_labels)
                axs[idx].set_xlabel("Time (h)")
            #axs[idx].set_ylabel(f"{mode_label}")

        # Add a single colorbar for all heatmaps
        fig.colorbar(cax, ax=axs, orientation="vertical", label=colorlabel)
        plt.show()