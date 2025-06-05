"""
Diagnostic plotting functions for SBIPIX.

This module contains plotting functions that work with SBIPIX class instances
to create visualizations for model diagnostics and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm
import dense_basis as db

from ..utils.sed_utils import mag_conversion


def plot_filters(sbipix_model):
    """
    Plot the filter transmission curves.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance with filter configuration
    """
    db.plot_filterset(
        filter_list=sbipix_model.filter_list, 
        filt_dir=sbipix_model.filter_path
    )


def plot_theta(sbipix_model, bins=50, limit_sfr=False, range_sfr=(-10, 2), 
               figsize=(12, 8), save=False, filename=None):
    """
    Plot histograms of the physical parameters.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance with simulation data
    bins : int, optional
        Number of histogram bins (default: 50)
    limit_sfr : bool, optional
        Whether to limit SFR histogram range (default: False)
    range_sfr : tuple, optional
        Range for SFR histogram if limit_sfr=True (default: (-10, 2))
    figsize : tuple, optional
        Figure size (default: (12, 8))
    save : bool, optional
        Whether to save plots (default: False)
    filename : str, optional
        Base filename for saving (default: None)
    """
    if sbipix_model.theta is None:
        raise ValueError("No simulation data found. Run load_simulation() first.")
    
    n_params = len(sbipix_model.theta[0, :])
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for i in range(n_params):
        ax = axes[i]
        ix = 2 if sbipix_model.both_masses else 1
        
        if i == ix and limit_sfr:
            ax.hist(sbipix_model.theta[:, i], bins=bins, range=range_sfr, 
                   alpha=0.7, edgecolor='black')
        else:
            ax.hist(sbipix_model.theta[:, i], bins=bins, 
                   alpha=0.7, edgecolor='black')
        
        ax.set_xlabel(sbipix_model.labels[i], fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(sbipix_model.theta[:, i])
        std_val = np.std(sbipix_model.theta[:, i])
        ax.text(0.05, 0.95, f'μ = {mean_val:.2f}\nσ = {std_val:.2f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        save_name = filename or 'parameter_histograms.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save_name}")
    
    plt.show()


def plot_fluxes(sbipix_model, n, figsize=(10, 6), save=False, filename=None,
                show_individual=True, show_median=True):
    """
    Plot the SEDs for the first n simulated galaxies.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance with simulation data
    n : int
        Number of galaxies to plot
    figsize : tuple, optional
        Figure size (default: (10, 6))
    save : bool, optional
        Whether to save plot (default: False)
    filename : str, optional
        Filename for saving (default: None)
    show_individual : bool, optional
        Whether to show individual SEDs (default: True)
    show_median : bool, optional
        Whether to show median SED (default: True)
    """
    if sbipix_model.obs is None:
        raise ValueError("No simulation data found. Run load_simulation() first.")
    
    fluxes = mag_conversion(sbipix_model.obs, convert_to='flux')
    lam_eff = np.load('./observational_properties/lam_eff.npy')[:]
    
    plt.figure(figsize=figsize)
    
    # Plot individual SEDs
    if show_individual:
        for i in range(min(n, len(fluxes))):
            plt.plot(lam_eff, fluxes[i], '.', alpha=0.6, markersize=4,
                    label=f'Galaxy {i+1}' if n <= 5 else None)
    
    # Plot median SED
    if show_median and n > 1:
        median_flux = np.median(fluxes[:n], axis=0)
        plt.plot(lam_eff, median_flux, 'ro-', linewidth=2, markersize=6,
                label=f'Median (n={n})')
    
    plt.yscale('log')
    plt.ylabel('Flux density [$\\mu$Jy]', fontsize=14)
    plt.xlabel('Wavelength [$\\AA$]', fontsize=14)
    plt.title(f'Simulated SEDs (showing {min(n, len(fluxes))} galaxies)', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    if (show_individual and n <= 5) or show_median:
        plt.legend()
    
    if save:
        save_name = filename or f'seds_n{n}.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save_name}")
    
    plt.show()


def plot_test_performance(sbipix_model, n_test=1000, n_theta=None, save=False, 
                         name=None, figsize=(8, 8)):
    """
    Plot test performance comparing true vs predicted values.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance with test results
    n_test : int, optional
        Number of test samples to plot (default: 1000)
    n_theta : int, optional
        Number of parameters to plot (default: auto-detect)
    save : bool, optional
        Whether to save plots (default: False)
    name : str, optional
        Custom filename prefix for saving (default: None)
    figsize : tuple, optional
        Figure size for each plot (default: (8, 8))
    """
    if sbipix_model.means_test is None or sbipix_model.theta is None:
        raise ValueError("No test results found. Run test_performance() first.")
    
    if n_theta is None:
        n_theta = 8 if sbipix_model.infer_z else 7

    for i in range(n_theta):
        p_true = sbipix_model.theta[:n_test, i]
        p_pred = sbipix_model.means_test[:, i]
        
        g = sns.jointplot(x=p_true, y=p_pred, height=figsize[0], 
                         label='Test data', legend=False)
        
        # Add perfect prediction line
        x_range = np.arange(np.min(p_true), np.max(p_true), 0.01)
        g.plot_joint(sns.kdeplot, color="k", zorder=1, levels=6, label='KDE')
        g.ax_joint.plot(x_range, x_range, '-r', label='y=x', linewidth=2)
        
        # Set labels
        g.set_axis_labels(
            sbipix_model.labels[i] + ' (true)', 
            sbipix_model.labels[i] + ' (predicted)', 
            fontsize=16
        )
        
        # Add R² score
        r2_score = sm.r2_score(p_true, p_pred)
        g.ax_joint.text(
            0.05, 0.95, f"R² = {r2_score:.3f}", 
            transform=g.ax_joint.transAxes,
            fontsize=14, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Add RMSE
        rmse = np.sqrt(np.mean((p_true - p_pred)**2))
        g.ax_joint.text(
            0.05, 0.88, f"RMSE = {rmse:.3f}", 
            transform=g.ax_joint.transAxes,
            fontsize=14,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        if save:
            if name is None:
                suffix = 'tau' if sbipix_model.parametric else 'dirichlet'
                filename = f'./plots/test_performance_{suffix}_{i}.pdf'
            else:
                filename = f'./plots/{name}{i}.pdf'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Saved plot to {filename}")
        
        plt.show()


def plot_performance_obs_from_catalog(sbipix_model, parameter='z', save=False, 
                                     filename=None, figsize=(8, 6)):
    """
    Plot performance on observational catalog.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance with observational results
    parameter : str, optional
        Which parameter to plot (default: 'z' for redshift)
    save : bool, optional
        Whether to save plot (default: False)
    filename : str, optional
        Custom filename (default: None)
    figsize : tuple, optional
        Figure size (default: (8, 6))
    """
    print(f"Plotting performance for parameter: {parameter}")
    
    plt.figure(figsize=figsize)
    
    # This would be implemented based on available observational data
    # Placeholder for now
    plt.text(0.5, 0.5, f'Observational performance plot for {parameter}\n'
                       'Implementation depends on available catalog data',
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    plt.title(f'Observational Performance: {parameter}', fontsize=16)
    
    if save:
        save_name = filename or f'obs_performance_{parameter}.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Plot saved as: {save_name}")
    
    plt.show()


def plot_corner(sbipix_model, n_samples=1000, parameters=None, save=False, 
                filename=None):
    """
    Create corner plot of posterior samples.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance
    n_samples : int, optional
        Number of samples to plot (default: 1000)
    parameters : list, optional
        Which parameters to include (default: None, uses all)
    save : bool, optional
        Whether to save plot (default: False)
    filename : str, optional
        Filename for saving (default: None)
    """
    try:
        import corner
    except ImportError:
        print("corner package not available. Install with: pip install corner")
        return
    
    if sbipix_model.theta is None:
        raise ValueError("No simulation data found. Run load_simulation() first.")
    
    # Select parameters to plot
    if parameters is None:
        data = sbipix_model.theta[:n_samples]
        labels = sbipix_model.labels
    else:
        param_indices = [sbipix_model.labels.index(p) for p in parameters]
        data = sbipix_model.theta[:n_samples, param_indices]
        labels = parameters
    
    # Create corner plot
    fig = corner.corner(data, labels=labels, show_titles=True, 
                       title_kwargs={"fontsize": 12})
    
    if save:
        save_name = filename or 'corner_plot.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Saved corner plot to {save_name}")
    
    plt.show()


def plot_sed_comparison(sbipix_model, galaxy_idx, model_sed=None, 
                       figsize=(10, 6), save=False, filename=None):
    """
    Compare observed and model SEDs for a specific galaxy.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance
    galaxy_idx : int
        Index of galaxy to plot
    model_sed : np.ndarray, optional
        Model SED to compare (default: None)
    figsize : tuple, optional
        Figure size (default: (10, 6))
    save : bool, optional
        Whether to save plot (default: False)
    filename : str, optional
        Filename for saving (default: None)
    """
    if sbipix_model.obs is None:
        raise ValueError("No simulation data found. Run load_simulation() first.")
    
    lam_eff = np.load('./observational_properties/lam_eff.npy')[:]
    obs_flux = mag_conversion(sbipix_model.obs[galaxy_idx], convert_to='flux')
    
    plt.figure(figsize=figsize)
    
    # Plot observed SED
    plt.errorbar(lam_eff, obs_flux, fmt='o', markersize=6, 
                label='Simulated Data', capsize=3)
    
    # Plot model SED if provided
    if model_sed is not None:
        plt.plot(lam_eff, model_sed, 's-', linewidth=2, markersize=4,
                label='Best-fit Model')
    
    plt.yscale('log')
    plt.xlabel('Wavelength [Å]', fontsize=14)
    plt.ylabel('Flux density [μJy]', fontsize=14)
    plt.title(f'SED Comparison - Galaxy {galaxy_idx}', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save:
        save_name = filename or f'sed_comparison_gal{galaxy_idx}.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save_name}")
    
    plt.show()

    def ppc_sed_comparison():
        """
        Placeholder for posterior predictive checks on SEDs.
        This function would implement the logic to compare observed SEDs
        with model predictions, typically using posterior samples.
        """
        print("Posterior predictive checks for SEDs not implemented yet.")
        # This would involve sampling from the posterior and comparing
        # the resulting SEDs with the observed data.
        # Implementation would depend on the specific model and data structure.
        # For now, we just print a message.
        # In a complete implementation, this would generate plots similar to
        # plot_sed_comparison but using posterior samples instead of a single model SED.
        pass