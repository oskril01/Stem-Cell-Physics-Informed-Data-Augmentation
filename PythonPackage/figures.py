import numpy as np
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wilcoxon

def batch_size_plotting(validation, importance, batches_validation, batches_importance, scale = 0.75, size = (10, 5), wilx = False):

    #Extract the MSE for the validation data
    mse_VCD = validation[:,:,:,:,:3,1,0]    
    mse_VCD_mean = np.mean(mse_VCD, axis = (2,3,4))

    mse_Pluri = validation[:,:,:,:,:3,1,1]
    mse_Pluri_mean = np.mean(mse_Pluri, axis = (2,3,4))

    #Means
    def get_means(data):
        #Remove all zero entries
        data = data[data != 0]
        means = np.mean(data, axis = 0)
        stds = np.std(data, axis = 0)
        return means, stds
    vcd_means = []
    pluri_means = []
    for i in range(3):
        vcd_means.append(get_means(mse_VCD_mean[:,i]))
        pluri_means.append(get_means(mse_Pluri_mean[:,i]))

    #T-Tests
    pairs = [(0, 1), (0, 2), (1, 2)]
    p_vals_vcd = {}
    p_vals_pluri = {}
    for i, j in pairs:
        p_vals_vcd[(i, j)] = stats.ttest_rel(mse_VCD[:,i].flatten(), mse_VCD[:,j].flatten())[1]
        p_vals_pluri[(i, j)] = stats.ttest_rel(mse_Pluri[:,i].flatten(), mse_Pluri[:,j].flatten())[1]
    print(p_vals_vcd)
    print(p_vals_pluri)


    #Prepare the importance values
    # delta_importances = np.zeros(((len(batches_importance)-1),3,2,4,2,20))
    delta_importances = np.zeros(((len(batches_importance)-1),3,2,3,2,20))
    for i in range(len(batches_importance)-1):
        delta_importances[i] = np.abs(importance[i] - importance[-1])
    # delta_importances = delta_importances[:,:,:,:3]
    delta_importances = delta_importances.mean(axis = (2,3,4,5))

    #Curve fitting
    def power_law(x, a, b):
        return a * x**b
    x_all = np.tile(batches_importance[:-1], 3)
    y_all = np.array([delta_importances[:,0], delta_importances[:,1], delta_importances[:,2]]).flatten()
    popt, pcov = curve_fit(power_law, x_all, y_all)
    x_fit = np.linspace(10, 750, 100)
    y_fit = power_law(x_fit, *popt)

    #print the power law fit eqn
    print(popt)

    #Print the R2 value
    residuals = y_all - power_law(x_all, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_all - np.mean(y_all))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(r_squared)

    #Plotting
    fig = plt.figure(figsize=size, dpi=300)
    sns.set_context("talk", font_scale=scale)

    # Create the main 1x2 grid (left: 2x2 plots, right: single plot)
    outer_gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2)
    inner_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[0], wspace=0.1, hspace=0.1, width_ratios=[3, 2])

    ax1 = fig.add_subplot(inner_gs[0, 0])  # Top-left
    ax2 = fig.add_subplot(inner_gs[1, 0])  # Bottom-left
    ax4 = fig.add_subplot(inner_gs[0, 1])  # Top-middle
    ax5 = fig.add_subplot(inner_gs[1, 1])  # Bottom-middle
    ax3 = fig.add_subplot(outer_gs[1])  # Entire right column

    colors = ['#FFC857','#E9724C','#C5283D']
    labels = ['fpAM', 'hdAM', 'nodeAM']


    #Validation Data
    for i in range(3):
        ax1.plot(batches_validation, mse_VCD_mean[:,i], color = colors[i], label = labels[i], marker = 'o')
        ax2.plot(batches_validation, mse_Pluri_mean[:,i], color = colors[i], label = labels[i], marker = 'o')

    ax1.set_title('Validation log-MSE')
    ax1.set_ylabel('VCD')
    ax1.set_xlabel('')
    ax1.set_xticks([])
    ax1.set_yscale('log')
    ax1.set_ylim(10e7, 10**12.25)

    ax2.set_ylabel('Pluripotency')
    ax2.set_xlabel('Batch Size $(N)$')
    ax2.set_yscale('log')
    ax2.set_ylim(10e-6, 10**(-0.75))

    
    # Means and T-Tests
    for i in range(3):
        ax4.bar(i, vcd_means[i][0], color = colors[i], label = labels[i], yerr = vcd_means[i][1])
        ax5.bar(i, pluri_means[i][0], color = colors[i], label = labels[i], yerr = pluri_means[i][1])

    ax4.set_title('Model Means')
    ax4.set_xticks([])
    ax4.set_yscale('log')

    ax5.set_xticks(np.arange(3))
    ax5.set_xticklabels(['fpAM', 'hdAM', 'nodeAM'])
    ax5.set_yscale('log')

    #T-Tests
    def significance_stars(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
        
    # Positioning for significance bars
    y_max_vcd = max([vcd_means[i][0] + vcd_means[i][1] for i in range(3)])
    base_bar_height_vcd = y_max_vcd * 1.05
    line_spacing_vcd = y_max_vcd * 0.08
    line_height_vcd = y_max_vcd * 0.02
    significance_levels_vcd = 0

    # Loop through model comparisons
    for (i, j) in pairs:
        p_value = p_vals_vcd[(i, j)]
        sig = significance_stars(p_value)

        if sig:
            y_position = base_bar_height_vcd + (significance_levels_vcd * line_spacing_vcd)
            ax4.plot([i, j], [y_position, y_position], color='black', lw=1)
            ax4.text((i + j) / 2, y_position + line_height_vcd, sig, ha='center', fontsize=12)
            significance_levels_vcd += 1

    # Positioning for significance bars
    y_max_pluri = max([pluri_means[i][0] + pluri_means[i][1] for i in range(3)])
    base_bar_height_pluri = y_max_pluri * 1.05
    line_spacing_pluri = y_max_pluri * 20
    line_height_pluri = y_max_pluri * 0.02
    significance_levels_pluri = 0

    # Loop through model comparisons
    for (i, j) in pairs:
        p_value = p_vals_pluri[(i, j)]
        sig = significance_stars(p_value)

        if sig:
            y_position = base_bar_height_pluri * (10 ** (significance_levels_pluri * line_spacing_pluri))  # Adjust for log scale
            ax5.plot([i, j], [y_position, y_position], color='black', lw=1)
            ax5.text((i + j) / 2, y_position * (10 ** line_height_pluri), sig, ha='center', fontsize=12)  # Adjust text positioning
            significance_levels_pluri += 1
    
    ax4.set_ylim(10e7, 10**12.25)
    ax5.set_ylim(10e-6, 10**(-0.75))
    ax4.set_yticks([])
    ax5.set_yticks([])


    #Importance Data
    for i in range(3):
        ax3.plot(batches_importance[:-1], delta_importances[:,i], color = colors[i], label = labels[i], marker = 'o', alpha = 0.75)
    ax3.plot(x_fit, y_fit, color = 'black', label = 'Power Law', linestyle = '-')
    ax3.set_title('Convergence of Importance Tensor')
    ax3.set_ylabel(r'$\|\mathcal{T}_N - \mathcal{T}_{1000} \|$')
    ax3.set_xlabel('Batch Size $(N)$')
    ax3.legend()

    # #Add text that shows the power law fit equation
    # ax3.text(0.5, 0.5, r'$\| \mathcal{T}_N - \mathcal{T}_{1000} \| \approx' + f" {popt[0]:.2f} N^{{{popt[1]:.2f}}}$", 
    #          horizontalalignment='center',
    #          verticalalignment='center',
    #             transform=ax3.transAxes,
    #             fontsize=12)


    plt.tight_layout()
    plt.show()

def data_with_heatmaps(
        Tensor, 
        mainData, 
        cell_type, 
        size = (12, 5), 
        scale = 0.75, 
        ratio = [0.5, 1, 0.5], 
        colormap_glucose = 'Blues',
        colormap_lactate = 'Reds', 
        color_glc = 'darkblue',
        color_lac = 'darkred',
        ticks = False
        ):
    
    if cell_type == 'H':
        mainData = mainData.Hdata
        cell = 0
    elif cell_type == 'I':
        mainData = mainData.Idata
        cell = 1
    else:
        return print('Choose a valid cell type: H or I')

    # Create figure and axis grid
    fig, axs = plt.subplots(3, 3, figsize=(size), dpi=300)
    sns.set_context("talk", font_scale=scale)
    gs = gridspec.GridSpec(3, 3, height_ratios=ratio, hspace = 0)

    titles = ['1.0 mM', '5.0 mM', '17.5 mM', '20.0 mM']

    for i in range(3):

        # Time for all
        time = np.linspace(0, len(mainData[i, 3, :] if i == 3 else mainData[i, 3, 1:]), len(mainData[i, 3, :] if i == 3 else mainData[i, 3, 1:]))
        time_labels = [str(int(i)) for i in time]

        glc_image = np.array([Tensor[0, cell, i, 0, :], Tensor[1, cell, i, 0, :], Tensor[2, cell, i, 0, :]])
        lac_image = np.array([Tensor[0, cell, i, 1, :], Tensor[1, cell, i, 1, :], Tensor[2, cell, i, 1, :]])

        # Data Trajectories
        ax1 = plt.subplot(gs[1, i])
        for j in range(1, 2*len(time_labels)):
            ax1.axvline(x=(j-1)/2, color='black', linestyle='-', alpha=0.15)
        # for j in range(1, len(time_labels)+1):
        #     ax1.axvline(x=j-1, color='black', linestyle='-', alpha=0.15)
        ax1.plot(mainData[i, 3, :] if i == 3 else mainData[i, 3, 1:], marker='.', color=color_glc, label = 'Glucose')
        ax1.plot(mainData[i, 4, :] if i == 3 else mainData[i, 4, 1:], marker='.', color=color_lac, label = 'Lactate')
        
        #add xticks below the plot with no labels
        if ticks:
            ax1.set_xticks(np.linspace(0, len(time_labels)-1, num=len(time_labels)))
            ax1.set_xticklabels([])
            ax1_top = ax1.twiny()
            ax1_top.set_xlim(ax1.get_xlim())
            ax1_top.set_xticks(np.linspace(0, len(time_labels)-1, num=len(time_labels)))
            ax1_top.set_xticklabels([])
        else:
            ax1.set_xticks([])
        ax1.set_ylim((-1, 26))
        if i == 0:
            ax1.set_ylabel('Concentration (mM)')
            ax1.legend(loc = 'upper left')
        else:
            ax1.set_yticks([])
        ax1.set_xlim(0, len(time_labels)-1)

        # Importance Heatmaps Glucose        
        ax0 = plt.subplot(gs[0, i])
        ax0.set_title(titles[i])
        rows, cols = glc_image.shape
        for r in range(1, rows):  # Avoid the first row (0) since the outline exists
            ax0.hlines(y=r - 0.5, xmin=-0.5, xmax=cols - 0.5, colors='black', linewidth=ax0.spines['left'].get_linewidth())
        im = ax0.imshow(glc_image, aspect='auto', cmap=colormap_glucose, vmin=0, vmax=1)
        ax0.xaxis.set_label_position('top')
        ax0.xaxis.tick_top()  # Move ticks to the top
        # ax0.set_xticks(np.linspace(0, glc_image.shape[1] - 1, num=len(time_labels)))  # Tick positions
        ax0.set_xticks([])
        # ax0.set_xticklabels([str(i) for i in np.linspace(0, len(time_labels) - 1, num=len(time_labels), dtype=int)])  # Tick labels
        ax0.set_xticklabels([])
        # ax0.set_xlabel('Time (days)')
        if i == 0:
            ax0.set_yticks(np.arange(3))  # Set y-tick positions at indices [0, 1, 2]
            ax0.set_yticklabels(['fpAM', 'hdAM', 'nodeAM'], rotation=0)
        else:
            ax0.set_yticks([])
        if i == 2:
            cbar_ax = fig.add_axes([1, 0.55, 0.01, 0.375]) # [1, 0.1, 0.01, 0.45]
            fig.colorbar(im, cax=cbar_ax, label='Glucose Imp')


        # Importance Heatmaps Lactate
        ax2 = plt.subplot(gs[2, i])
        im = ax2.imshow(lac_image, aspect='auto', cmap=colormap_lactate, vmin=0, vmax=1)
        rows, cols = lac_image.shape
        for r in range(1, rows):  # Avoid the first row (0) since the outline exists
            ax2.hlines(y=r - 0.5, xmin=-0.5, xmax=cols - 0.5, colors='black', linewidth=ax2.spines['left'].get_linewidth())
        ax2.set_xticks(np.linspace(0, cols - 1, num=len(time_labels)))
        ax2.set_xticklabels(np.linspace(0, len(time_labels) - 1, num=len(time_labels), dtype=int))
        ax2.set_xlabel('Time (days)')
        if i == 0:
            ax2.set_yticks(np.arange(3))
            ax2.set_yticklabels(['fpAM', 'hdAM', 'nodeAM'], rotation=0)
        else:
            ax2.set_yticks([])
        if i == 2:
            cbar_ax = fig.add_axes([1, 0.1, 0.01, 0.4])
            fig.colorbar(im, cax=cbar_ax, label='Lactate Imp')

    # Remove all ticks and labels
    for ax in axs.flat:
        ax.label_outer()
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def development_figure(
        index,
        mainData,
        size = (12, 6),
        scale = 0.75,
        fpm_color = '#FFC857',
        hdm_color = '#E9724C',
        node_color = '#C5283D',
        opacity = 0.5
):

    'Helper Function'
    def get_95_CI(data):
        n = data.shape[0]
        mean = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)
        lower = mean - 1.96*std
        upper = mean + 1.96*std

        for i in range(np.shape(lower)[0]):
            for j in range(np.shape(lower)[1]):
                if lower[i,j] < 0:
                    lower[i,j] = 0
        
        return lower, upper


    'DATA PREPARATION'

    #Get the correct experimental data
    if index[0] == 0:
        if index[1] == 3:
            exp_data = mainData.Hdata[index[1]]
        else:
            exp_data = mainData.Hdata[index[1]][:,1:]
    elif index[0] == 1:
        if index[1] == 3:
            exp_data = mainData.Idata[index[1]]
        else:
            exp_data = mainData.Idata[index[1]][:,1:]

    #Get the Augmented Data
    if index[0] == 0:
        fpam_aug = mainData.hcell_FPAMs[index[1]].augmented_data
        hdam_aug = mainData.hcell_HDAMs[index[1]].augmented_data
        nodeam_aug = mainData.hcell_NODEAMs[index[1]].augmented_data
    elif index[0] == 1:
        fpam_aug = mainData.icell_FPAMs[index[1]].augmented_data
        hdam_aug = mainData.icell_HDAMs[index[1]].augmented_data
        nodeam_aug = mainData.icell_NODEAMs[index[1]].augmented_data

    #Get the 95% confidence intervals
    lower_fpam, upper_fpam = get_95_CI(fpam_aug)
    lower_hdam, upper_hdam = get_95_CI(hdam_aug)
    lower_nodeam, upper_nodeam = get_95_CI(nodeam_aug)

    #Get the timepoints
    days = np.shape(exp_data)[1]
    data_time = np.linspace(0, 24*days, days)
    aug_data_time = np.linspace(0, 24*days, 20)

    xaxis = np.linspace(0, days - 1, days)
    xaxis_labels = [str(int(i)) for i in xaxis]

    "FIGURE"

    #Layout the figure
    fig = plt.figure(figsize=size, dpi = 300)
    sns.set_context("talk", font_scale=scale)

    #Set up the grids
    outer_grid = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.15)
    left_grid = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_grid[0], wspace=0.05, hspace=0.05)
    right_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[1], wspace=0.05, hspace=0.05)

    # Create subplots for the left grid
    left_axes_row1 = [fig.add_subplot(left_grid[0, i]) for i in range(3)]  # First row, independent shared y-axis
    left_axes_row2 = [fig.add_subplot(left_grid[1, i]) for i in range(3)]  # Second row, independent shared y-axis

    #Shared axis
    for i in range(1, 3):
        left_axes_row1[i].sharey(left_axes_row1[0])
    for i in range(1, 3):
        left_axes_row2[i].sharey(left_axes_row2[0])

    # Combine both rows into a single list
    left_axes = [left_axes_row1, left_axes_row2]

    # Create subplots for the right grid
    right_axes_row1 = [fig.add_subplot(right_grid[0, i]) for i in range(2)]  # First row, independent shared y-axis
    right_axes_row2 = [fig.add_subplot(right_grid[1, i]) for i in range(2)]  # Second row, independent shared y-axis

    # Ensure the first row shares y-axes within itself
    for i in range(1, 2):
        right_axes_row1[i].sharey(right_axes_row1[0])

    # Ensure the second row shares y-axes within itself
    for i in range(1, 2):
        right_axes_row2[i].sharey(right_axes_row2[0])

    # Combine both rows into a single list
    right_axes = [right_axes_row1, right_axes_row2]


    'Plotting'

    # Plot data in the left grid (Xv, Xt, NAN+)
    left_labels = [r'$X_v$', r'$X_t$', r'$X_{NAN^{+}}$']
    for i in range(3):
        for j in range(2):
            ax = left_axes[j][i]

            #set the ylimits for the right grid
            ax.set_ylim(-.1, 1.1)

            # First row: Plot actual data
            if j == 0:
                sns.scatterplot(x=data_time, y=exp_data[i] / 1e6, color='black', ax=ax, zorder=3)
                sns.lineplot(x=aug_data_time, y=fpam_aug[0, i, :].T/1e6, ax=ax, color=fpm_color, label='fpAM')
                sns.lineplot(x=aug_data_time, y=hdam_aug[0, i, :].T/1e6 if i != 2 else hdam_aug[0,4,:]/1e6, ax=ax, color=hdm_color, label='hdAM')
                sns.lineplot(x=aug_data_time, y=nodeam_aug[0, i, :].T/1e6, ax=ax, color=node_color, label='nodeAM')
            
                if i == 0:
                    #show the legend
                    ax.legend(loc='upper left', fontsize='small')

                # if i == 0:
                #     #create dummy lines for the legend
                #     dummy0 = mlines.Line2D([], [], color='black', label='Exp Data', marker='o', linestyle='None')
                #     dummy1 = mlines.Line2D([], [], color='black', label='Optimal Fit')
                #     dummy2 = ax.fill_between([], [], [], color='black', alpha=opacity, label='Augmented Data')
                #     ax.legend(handles=[dummy0, dummy1, dummy2], loc='upper left', fontsize='small')
            
            # Second row: Plot confidence intervals
            else:
                ax.fill_between(aug_data_time, lower_fpam[0]/1e6 if i == 0 else lower_fpam[1]/1e6 if i == 1 else lower_fpam[2]/1e6, 
                                upper_fpam[0]/1e6 if i == 0 else upper_fpam[1]/1e6 if i == 1 else upper_fpam[2]/1e6, 
                                color=fpm_color, alpha=opacity, label='fpAM')
                ax.fill_between(aug_data_time, lower_hdam[0]/1e6 if i == 0 else lower_hdam[1]/1e6 if i == 1 else lower_hdam[4]/1e6, 
                                upper_hdam[0]/1e6 if i == 0 else upper_hdam[1]/1e6 if i == 1 else upper_hdam[4]/1e6, 
                                color=hdm_color, alpha=opacity, label='hdAM')
                ax.fill_between(aug_data_time, lower_nodeam[0]/1e6 if i == 0 else lower_nodeam[1]/1e6 if i == 1 else lower_nodeam[2]/1e6,
                                upper_nodeam[0]/1e6 if i == 0 else upper_nodeam[1]/1e6 if i == 1 else upper_nodeam[2]/1e6,
                                color=node_color, alpha=opacity, label='nodeAM') 
                sns.lineplot(x=aug_data_time, y=fpam_aug[0, i, :].T/1e6, ax=ax, color=fpm_color)
                sns.lineplot(x=aug_data_time, y=hdam_aug[0, i, :].T/1e6 if i != 2 else hdam_aug[0,4,:]/1e6, ax=ax, color=hdm_color)
                sns.lineplot(x=aug_data_time, y=nodeam_aug[0, i, :].T/1e6, ax=ax, color=node_color)
                ax.get_yaxis().get_offset_text().set_visible(False)  # This hides the '1e6' label
            ax.set_title(left_labels[i] if j == 0 else "", fontsize='large')
            ax.set_xticklabels([]) if j == 0 else ax.set_xlabel("Time (days)")
            if j ==0:
                ax.tick_params(axis='x', length=0)# This removes x-axis tick marks
            if j == 1:
                #label the x-axis with xaxis and xaxis_labels
                ax.set_xticks(data_time)
                ax.set_xticklabels(xaxis_labels)
            if i == 0:
                quest=0
                #ax.set_ylabel(r"Cell Density $\left(\frac{\mathrm{cells} \times 10^6}{\mathrm{mL}}\right)$")
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)

            #Only keep top left legend
            if i != 0 or j != 0:
                ax.get_legend().remove()
            # ax.get_legend().remove()

    # Plot data in the right grid (GLC, LAC)
    right_labels = [r'$GLC$', r'$LAC$']
    for i in range(2):
        for j in range(2):
            ax = right_axes[j][i]

            #set the ylimits for the right grid
            ax.set_ylim(-1, 16)

            # First row: Plot actual data
            if j == 0: 
                sns.scatterplot(x=data_time, y=exp_data[i + 3], color='black', ax=ax, zorder=3)
                sns.lineplot(x=aug_data_time, y=fpam_aug[0, i + 3, :].T, ax=ax, color=fpm_color, label='fpAM')
                sns.lineplot(x=aug_data_time, y=hdam_aug[0, i + 2, :].T, ax=ax, color=hdm_color, label='hdAM')
                sns.lineplot(x=aug_data_time, y=nodeam_aug[0, i + 3, :].T, ax=ax, color=node_color, label='nodeAM')
            
            # Second row: Plot confidence intervals
            else:
                ax.fill_between(aug_data_time, lower_fpam[3] if i == 0 else lower_fpam[4], 
                                upper_fpam[3] if i == 0 else upper_fpam[4], 
                                color=fpm_color, alpha=opacity, label='fpAM')
                ax.fill_between(aug_data_time, lower_hdam[2] if i == 0 else lower_hdam[3],
                                upper_hdam[2] if i == 0 else upper_hdam[3], 
                                color=hdm_color, alpha=opacity, label='hdAM')
                ax.fill_between(aug_data_time, lower_nodeam[3] if i == 0 else lower_nodeam[4],
                                upper_nodeam[3] if i == 0 else upper_nodeam[4],
                                color=node_color, alpha=opacity, label='nodeAM')
                
                sns.lineplot(x=aug_data_time, y=fpam_aug[0, i + 3, :].T, ax=ax, color=fpm_color)
                sns.lineplot(x=aug_data_time, y=hdam_aug[0, i + 2, :].T, ax=ax, color=hdm_color)
                sns.lineplot(x=aug_data_time, y=nodeam_aug[0, i + 3, :].T, ax=ax, color=node_color)

            ax.set_title(right_labels[i] if j == 0 else "", fontsize='large')
            ax.set_xticklabels([]) if j == 0 else ax.set_xlabel("Time (days)")
            if j ==0:
                ax.tick_params(axis='x', length=0)# This removes x-axis tick marks
            if j == 1:
                #label the x-axis with xaxis and xaxis_labels
                ax.set_xticks(data_time)
                ax.set_xticklabels(xaxis_labels)
            if i == 0:
                quest2 = 1 #ax.set_ylabel("Concentration (mM)")
                # ax.set_ylabel("hold", color = 'black')
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)
            # remove all legends
            # if i == 0: #and j == 0:
            #     ax.legend(loc='upper left', fontsize='small')
            # else:
            #     ax.get_legend().remove()
            ax.get_legend().remove()

    # Add a single y-axis label for each side, positioned between rows
    fig.text(0.06, 0.5, r'Cell Density $\left(\frac{\mathrm{cells} \times 10^6}{\mathrm{mL}}\right)$', va='center', rotation='vertical', fontsize='medium')
    fig.text(0.57, 0.5, 'Concentration (mM)', va='center', rotation='vertical', fontsize='medium')

    plt.show()

    return None

def model_accuracy(
    mainData,
    fpm_color='#FFC857',
    hdm_color='#E9724C',
    node_color='#C5283D',
    scale=0.75,
    size=(12, 3),
    wilx=False
):
    loss_tensor = mainData.get_loss_values()
    
    # Labels and shapes
    models, cells, datasets = np.shape(loss_tensor)
    model_labels = ['fpAM', 'hdAM', 'nodeAM']
    dataset_labels = ['1.0', '5.0', '17.5', '20.0']

    # Reshape data
    low_data = loss_tensor[:, :, :2]
    high_data = loss_tensor[:, :, 2:]

    model_data_low = np.reshape(low_data, (models, cells * 2))
    model_data_high = np.reshape(high_data, (models, cells * 2))

    pairs = [(0, 1), (0, 2), (1, 2)]
    low_p_vals = {}
    high_p_vals = {}

    for i, j in pairs:
        if wilx:
            low_p_vals[(i, j)] = wilcoxon(model_data_low[i, :], model_data_low[j, :])[1]
            high_p_vals[(i, j)] = wilcoxon(model_data_high[i, :], model_data_high[j, :])[1]
        else:
            low_p_vals[(i, j)] = stats.ttest_rel(model_data_low[i, :], model_data_low[j, :])[1]
            high_p_vals[(i, j)] = stats.ttest_rel(model_data_high[i, :], model_data_high[j, :])[1]

    low_p_vals = {key: round(value, 4) for key, value in low_p_vals.items()}
    high_p_vals = {key: round(value, 4) for key, value in high_p_vals.items()}

    print(low_p_vals)
    print(high_p_vals)

    # Set Seaborn context for style
    sns.set_context("talk", font_scale=scale)
    fig, axs = plt.subplots(1, 3, figsize=size, dpi=300, gridspec_kw={'width_ratios': [4, 4, 2]}, sharey=True)

    # Colors for the models
    colors = [fpm_color, hdm_color, node_color]

    # Subplot 1: hESCs
    axs[0].set_title('hESCs')
    axs[0].set_ylabel(r'Optimal Loss $\mathcal{L}(p^*)$')
    for i in range(models):
        axs[0].bar(np.arange(datasets) + (i - 1) * 0.2, loss_tensor[i, 0, :], width=0.2, color=colors[i], label=model_labels[i])
    axs[0].set_xticks(np.arange(datasets))
    axs[0].set_xticklabels(dataset_labels)
    axs[0].legend()
    axs[0].set_xlabel('Glucose Feed (mM)')

    # Subplot 2: hiPSCs
    axs[1].set_title('hiPSCs')
    for i in range(models):
        axs[1].bar(np.arange(datasets) + (i - 1) * 0.2, loss_tensor[i, 1, :], width=0.2, color=colors[i], label=model_labels[i])
    axs[1].set_xticks(np.arange(datasets))
    axs[1].set_xticklabels(dataset_labels)
    axs[1].set_xlabel('Glucose Feed (mM)')

    # Compute means and standard deviations
    mean_loss_lowglucose = np.mean(loss_tensor[:, :, :2], axis=2)
    std_loss_lowglucose = np.std(loss_tensor[:, :, :2], axis=2)
    mean_loss_highglucose = np.mean(loss_tensor[:, :, 2:], axis=2)
    std_loss_highglucose = np.std(loss_tensor[:, :, 2:], axis=2)

    # Adjust final subplot (Means) to center the bars
    x = np.arange(2) # Two conditions: Low and High Glucose
    width = 0.25  # Bar width
    offsets = np.linspace(-width, width, models)  # Create symmetric offsets

    axs[2].set_title('Pooled Means')
    # Loop over models and plot each as a separate set of bars
    for i in range(models):
        axs[2].bar(x + offsets[i], [mean_loss_lowglucose[i, 0], mean_loss_highglucose[i, 0]], 
                yerr=[std_loss_lowglucose[i, 0], std_loss_highglucose[i, 0]], 
                width=width, color=colors[i], alpha=0.75, label=model_labels[i])
    # Set correct x-axis labels
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(['Low', 'High'])
    axs[2].set_xlabel('Glucose Feed Group')

       # Function to determine significance labels
    def significance_stars(p):
        if p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    # Set y-axis limit to ensure space for annotations
    y_max = max(np.max(mean_loss_lowglucose[:, 0] + std_loss_lowglucose[:, 0]),
                np.max(mean_loss_highglucose[:, 0] + std_loss_highglucose[:, 0]))

    axs[2].set_ylim(0, y_max * 1.3)  # Extend y-limit to fit annotations

    # Position for significance bars
    base_bar_height = y_max * 1.05  # Initial height for bars
    line_spacing = y_max * 0.08  # Vertical spacing between multiple significance lines
    line_height = y_max * 0.02  # Small vertical tick height

    # Track how many significance bars are placed at each x position
    low_significance_levels = 0
    high_significance_levels = 0

    # Loop through model comparisons
    for (i, j) in pairs:
        low_p = low_p_vals[(i, j)]
        high_p = high_p_vals[(i, j)]
        
        low_significance = significance_stars(low_p)
        high_significance = significance_stars(high_p)

        # Position for Low Glucose (x=0)
        if low_significance:
            y_position = base_bar_height + (low_significance_levels * line_spacing)
            axs[2].plot([x[0] + offsets[i], x[0] + offsets[j]], [y_position, y_position], color='black', lw=1)
            axs[2].text((x[0] + offsets[i] + x[0] + offsets[j]) / 2, y_position + line_height, low_significance, 
                        ha='center', fontsize=12)
            low_significance_levels += 1  # Increment to avoid overlap

        # Position for High Glucose (x=1)
        if high_significance:
            y_position = base_bar_height + (high_significance_levels * line_spacing)
            axs[2].plot([x[1] + offsets[i], x[1] + offsets[j]], [y_position, y_position], color='black', lw=1)
            axs[2].text((x[1] + offsets[i] + x[1] + offsets[j]) / 2, y_position + line_height, high_significance, 
                        ha='center', fontsize=12)
            high_significance_levels += 1  # Increment to avoid overlap

    # Add indicator for Low and High Glucose below bars

    # Final layout adjustments
    plt.tight_layout()
    plt.show()
