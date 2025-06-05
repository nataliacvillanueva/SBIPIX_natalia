#!/usr/bin/env python3
"""
JADES Galaxy HDF5 Data Inspector

This script provides tools to explore and visualize the six_galaxies_data.hdf5 file
containing JWST/HST photometry data for six sample JADES galaxies.

Usage:
    python inspect_jades_hdf5.py [filename]
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

class JADESInspector:
    """Class to inspect and visualize JADES galaxy HDF5 data"""
    
    def __init__(self, filename="six_galaxies_data.hdf5"):
        """
        Initialize the inspector
        
        Parameters:
        -----------
        filename : str
            Path to the HDF5 file
        """
        self.filename = filename
        self.check_file_exists()
        self.load_metadata()
    
    def check_file_exists(self):
        """Check if the HDF5 file exists"""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"HDF5 file '{self.filename}' not found!")
    
    def load_metadata(self):
        """Load metadata from the HDF5 file"""
        with h5py.File(self.filename, "r") as hdf:
            # Load filters and decode from bytes
            self.filters = [f.decode('utf-8') for f in hdf["metadata"]["filters"]]
            self.units = hdf["metadata"].attrs["units"]
            self.num_galaxies = hdf["metadata"].attrs["num_galaxies"]
            self.grid_shape = hdf["metadata"].attrs["coordinates grid"]
            self.galaxy_ids = list(hdf["galaxies"].keys())
    
    def print_overview(self):
        """Print a general overview of the dataset"""
        print("=" * 60)
        print("🌌 JADES Galaxy HDF5 Data Overview")
        print("=" * 60)
        print(f"📁 File: {self.filename}")
        print(f"🔢 Number of galaxies: {self.num_galaxies}")
        print(f"📐 Coordinate grid: {self.grid_shape}")
        print(f"📊 Units: {self.units}")
        print(f"🔍 Galaxy IDs: {', '.join(self.galaxy_ids)}")
        print(f"🔧 Number of filters: {len(self.filters)}")
        print()
        
        # Print filter information
        print("📡 Available Filters:")
        jwst_filters = [f for f in self.filters if f.startswith("JWST")]
        hst_filters = [f for f in self.filters if f.startswith("HST")]
        
        print(f"  JWST NIRCam ({len(jwst_filters)} filters):")
        for i, filt in enumerate(jwst_filters):
            print(f"    {i+1:2d}. {filt}")
        
        print(f"  HST ACS ({len(hst_filters)} filters):")
        for i, filt in enumerate(hst_filters):
            print(f"    {i+len(jwst_filters)+1:2d}. {filt}")
        print()
    
    def inspect_galaxy(self, galaxy_id):
        """
        Inspect a specific galaxy
        
        Parameters:
        -----------
        galaxy_id : str or int
            Galaxy ID to inspect
        """
        galaxy_id = str(galaxy_id)
        
        if galaxy_id not in self.galaxy_ids:
            print(f"❌ Galaxy ID '{galaxy_id}' not found!")
            print(f"Available IDs: {', '.join(self.galaxy_ids)}")
            return
        
        with h5py.File(self.filename, "r") as hdf:
            galaxy = hdf[f"galaxies/{galaxy_id}"]
            
            # Load data
            coordinates = galaxy["coordinates"][()]  # 2D array of pixel coordinates
            photometry = galaxy["photometry"][()]   # photometry data (num_pixels, num_filters)
            errors = galaxy["error"][()]             # photometric errors (num_pixels, num_filters)
            redshift = galaxy.attrs.get("redshift", "N/A")
            
            print(f"🌟 Galaxy {galaxy_id} Details:")
            print(f"  🔴 Redshift: {redshift:.2f}")
            print(f"  📍 Number of pixels: {len(coordinates)}")
            print(f"  📊 Photometry shape: {photometry.shape}")
            print(f"  🎯 Coordinate range: X=[{coordinates[:, 0].min():.1f}, {coordinates[:, 0].max():.1f}], "
                  f"Y=[{coordinates[:, 1].min():.1f}, {coordinates[:, 1].max():.1f}]")
            
            # Show statistics
            valid_phot = photometry[np.isfinite(photometry)]
            if len(valid_phot) > 0:
                print(f"  📈 Photometry range: [{valid_phot.min():.2e}, {valid_phot.max():.2e}] μJy")
                print(f"  🎲 Non-detection fraction: {np.sum(~np.isfinite(photometry)) / photometry.size:.1%}")
            
            return {
                'coordinates': coordinates,
                'photometry': photometry,
                'errors': errors,
                'redshift': redshift
            }
    
    def plot_galaxy_footprint(self, galaxy_id=None, figsize=(12, 8)):
        """
        Plot the spatial distribution of galaxy pixels
        
        Parameters:
        -----------
        galaxy_id : str, int, or None
            Specific galaxy to plot, or None for all galaxies
        figsize : tuple
            Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        with h5py.File(self.filename, "r") as hdf:
            
            if galaxy_id is not None:
                # Plot single galaxy
                galaxy_id = str(galaxy_id)
                if galaxy_id not in self.galaxy_ids:
                    print(f"❌ Galaxy ID '{galaxy_id}' not found!")
                    return
                
                galaxy = hdf[f"galaxies/{galaxy_id}"]
                coords = galaxy["coordinates"][()]
                redshift = galaxy.attrs.get("redshift", "N/A")
                
                ax1.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=20)
                ax1.set_title(f"Galaxy {galaxy_id} Footprint\n(z = {redshift:.2f})")
                
            else:
                # Plot all galaxies
                colors = plt.cm.Set3(np.linspace(0, 1, len(self.galaxy_ids)))
                
                for i, gal_id in enumerate(self.galaxy_ids):
                    galaxy = hdf[f"galaxies/{gal_id}"]
                    coords = galaxy["coordinates"][()]
                    redshift = galaxy.attrs.get("redshift", "N/A")
                    
                    ax1.scatter(coords[:, 0], coords[:, 1], 
                              alpha=0.7, s=20, color=colors[i], 
                              label=f"{gal_id} (z={redshift:.2f})")
                
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.set_title("All Galaxy Footprints")
            
            ax1.set_xlabel("X coordinate (pixels)")
            ax1.set_ylabel("Y coordinate (pixels)")
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, self.grid_shape[0])
            ax1.set_ylim(0, self.grid_shape[1])
            
            # Plot pixel count histogram
            pixel_counts = []
            gal_labels = []
            for gal_id in self.galaxy_ids:
                galaxy = hdf[f"galaxies/{gal_id}"]
                coords = galaxy["coordinates"][()]
                pixel_counts.append(len(coords))
                gal_labels.append(gal_id)
            
            bars = ax2.bar(gal_labels, pixel_counts, alpha=0.7)
            ax2.set_title("Number of Pixels per Galaxy")
            ax2.set_xlabel("Galaxy ID")
            ax2.set_ylabel("Number of Pixels")
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, pixel_counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pixel_counts)*0.01,
                        str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def find_max_snr_pixel(self, galaxy_id, filters_for_snr=['F277W', 'F356W', 'F444W']):
        """
        Find the pixel with maximum S/N averaged over specified filters
        
        Parameters:
        -----------
        galaxy_id : str
            Galaxy ID to analyze
        filters_for_snr : list
            List of filter names to use for S/N calculation
            
        Returns:
        --------
        max_snr_idx : int
            Index of pixel with maximum S/N
        max_snr_value : float
            Maximum S/N value
        """
        with h5py.File(self.filename, "r") as hdf:
            galaxy = hdf[f"galaxies/{galaxy_id}"]
            photometry = galaxy["photometry"][()]
            errors = galaxy["error"][()]
            
            # Find filter indices
            filter_indices = []
            for filt in filters_for_snr:
                for i, full_filter in enumerate(self.filters):
                    if filt in full_filter:
                        filter_indices.append(i)
                        break
            
            if len(filter_indices) == 0:
                print(f"❌ None of the filters {filters_for_snr} found in dataset!")
                return None, None
            
            print(f"📡 Using filters for S/N: {[self.filters[i] for i in filter_indices]}")
            
            # Calculate S/N for each pixel
            snr_values = []
            for pixel_idx in range(len(photometry)):
                pixel_snr = []
                for filt_idx in filter_indices:
                    flux = photometry[pixel_idx, filt_idx]
                    error = errors[pixel_idx, filt_idx]
                    
                    if np.isfinite(flux) and np.isfinite(error) and error > 0:
                        pixel_snr.append(flux / error)
                
                if len(pixel_snr) > 0:
                    snr_values.append(np.mean(pixel_snr))
                else:
                    snr_values.append(np.nan)
            
            snr_values = np.array(snr_values)
            
            # Find maximum S/N pixel
            valid_snr = snr_values[np.isfinite(snr_values)]
            if len(valid_snr) == 0:
                print("❌ No valid S/N values found!")
                return None, None
            
            max_snr_idx = np.nanargmax(snr_values)
            max_snr_value = snr_values[max_snr_idx]
            
            print(f"🎯 Maximum S/N pixel: {max_snr_idx} (S/N = {max_snr_value:.2f})")
            
            return max_snr_idx, max_snr_value
    
    def plot_sed(self, galaxy_id, pixel_idx=None, show_errors=True, show_upper_limits=None, figsize=(12, 6)):
        """
        Plot spectral energy distribution for a galaxy
        
        Parameters:
        -----------
        galaxy_id : str or int
            Galaxy ID to plot
        pixel_idx : int, str, or None
            Specific pixel index, 'max_snr' for max S/N pixel, or None for median SED
        show_errors : bool
            Whether to show error bars
        show_upper_limits : bool or None
            Whether to show upper limits, None to ask user
        figsize : tuple
            Figure size
        """
        galaxy_id = str(galaxy_id)
        
        if galaxy_id not in self.galaxy_ids:
            print(f"❌ Galaxy ID '{galaxy_id}' not found!")
            return
        
        # Ask about upper limits if not specified
        if show_upper_limits is None:
            response = input("Include upper limits? (y/n): ").strip().lower()
            show_upper_limits = response in ['y', 'yes', '1', 'true']
        
        # Load background noise limits if requested
        upper_limits = None
        if show_upper_limits:
            try:
                # Try to find the background noise file
                noise_file = os.path.dirname(os.path.abspath(__file__))+'/obs_properties/background_noise_hainline.npy'
                upper_limits = np.load(noise_file, allow_pickle=True)
                print(f"✅ Loaded upper limits from: {noise_file}")
                if len(upper_limits) != len(self.filters):
                    print(f"⚠️  Warning: Upper limits array length ({len(upper_limits)}) doesn't match filters ({len(self.filters)})")
            except Exception as e:
                print(f"❌ Error loading upper limits: {e}")
                show_upper_limits = False
        
        with h5py.File(self.filename, "r") as hdf:
            galaxy = hdf[f"galaxies/{galaxy_id}"]
            photometry = galaxy["photometry"][()]
            errors = galaxy["error"][()]
            redshift = galaxy.attrs.get("redshift", "N/A")
            
            # Effective wavelengths for plotting (approximate)
            jwst_waves = [0.90, 1.15, 1.50, 1.82, 2.00, 2.10, 2.77, 3.35, 3.56, 4.10, 4.30, 4.44, 4.60, 4.80]
            hst_waves = [0.435, 0.606, 0.775, 0.814, 0.850]
            wavelengths = np.array(jwst_waves + hst_waves)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            if pixel_idx == 'max_snr':
                # Find pixel with maximum S/N
                max_snr_idx, max_snr_value = self.find_max_snr_pixel(galaxy_id)
                if max_snr_idx is None:
                    return
                pixel_idx = max_snr_idx
                flux = photometry[pixel_idx]
                flux_err = errors[pixel_idx]
                title_suffix = f"Max S/N Pixel {pixel_idx} (S/N={max_snr_value:.2f})"
                
            elif pixel_idx is not None:
                # Plot specific pixel
                if pixel_idx >= len(photometry):
                    print(f"❌ Pixel index {pixel_idx} out of range (max: {len(photometry)-1})")
                    return
                
                flux = photometry[pixel_idx]
                flux_err = errors[pixel_idx]
                title_suffix = f"Pixel {pixel_idx}"
                
            else:
                # Plot median SED
                flux = np.nanmedian(photometry, axis=0)
                flux_err = np.nanmedian(errors, axis=0)
                title_suffix = "Median SED"
            
            # Separate JWST and HST points
            jwst_mask = np.arange(len(jwst_waves))
            hst_mask = np.arange(len(jwst_waves), len(wavelengths))
            
            # Plot SED
            if show_errors:
                ax1.errorbar(wavelengths[jwst_mask], flux[jwst_mask], yerr=flux_err[jwst_mask],
                           fmt='o', color='red', label='JWST NIRCam', capsize=3, alpha=0.8)
                ax1.errorbar(wavelengths[hst_mask], flux[hst_mask], yerr=flux_err[hst_mask],
                           fmt='s', color='blue', label='HST ACS', capsize=3, alpha=0.8)
            else:
                ax1.plot(wavelengths[jwst_mask], flux[jwst_mask], 'ro', label='JWST NIRCam', alpha=0.8)
                ax1.plot(wavelengths[hst_mask], flux[hst_mask], 'bs', label='HST ACS', alpha=0.8)
            
            # Plot upper limits if requested
            if show_upper_limits and upper_limits is not None:
                # Plot JWST upper limits
                ax1.scatter(wavelengths[jwst_mask], upper_limits[jwst_mask],
                           marker='v', color='red', alpha=0.6, s=50,
                           label='JWST Upper Limits')
                
                # Plot HST upper limits  
                ax1.scatter(wavelengths[hst_mask], upper_limits[hst_mask],
                           marker='v', color='blue', alpha=0.6, s=50,
                           label='HST Upper Limits')
            
            ax1.set_xlabel('Wavelength (μm)')
            ax1.set_ylabel('Flux (μJy)')
            title = f'Galaxy {galaxy_id} - {title_suffix}\n(z = {redshift:.2f})'
            if show_upper_limits:
                title += ' [with upper limits]'
            ax1.set_title(title)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot flux distribution across pixels for each filter
            valid_pixels = np.sum(np.isfinite(photometry), axis=0)
            ax2.bar(range(len(self.filters)), valid_pixels, alpha=0.7)
            ax2.set_xlabel('Filter Index')
            ax2.set_ylabel('Number of Valid Pixels')
            ax2.set_title('Valid Detections per Filter')
            
            # Add filter labels
            filter_labels = [f.split('_')[-1] for f in self.filters]
            ax2.set_xticks(range(len(self.filters)))
            ax2.set_xticklabels(filter_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def create_summary_table(self):
        """Create a summary table of all galaxies"""
        print("\n📋 Galaxy Summary Table:")
        print("-" * 80)
        print(f"{'ID':<8} {'Redshift':<10} {'Pixels':<8} {'Valid Phot':<12} {'Coord Range':<20}")
        print("-" * 80)
        
        with h5py.File(self.filename, "r") as hdf:
            for gal_id in self.galaxy_ids:
                galaxy = hdf[f"galaxies/{gal_id}"]
                coords = galaxy["coordinates"][()]
                phot = galaxy["photometry"][()]
                redshift = galaxy.attrs.get("redshift", "N/A")
                
                n_pixels = len(coords)
                valid_frac = np.sum(np.isfinite(phot)) / phot.size
                coord_range = f"({coords[:,0].min():.0f}-{coords[:,0].max():.0f}, {coords[:,1].min():.0f}-{coords[:,1].max():.0f})"
                
                print(f"{gal_id:<8} {redshift:<.2f} {n_pixels:<8} {valid_frac:<12.1%} {coord_range:<20}")
        
        print("-" * 80)

def interactive_explorer(filename="six_galaxies_data.hdf5"):
    """
    Interactive command-line explorer
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
    """
    try:
        inspector = JADESInspector(filename)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    inspector.print_overview()
    
    while True:
        print("\n🔍 What would you like to do?")
        print("1. Inspect specific galaxy")
        print("2. Plot galaxy footprints")
        print("3. Plot SED for a galaxy")
        print("4. Show summary table")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            gal_id = input(f"Enter galaxy ID ({', '.join(inspector.galaxy_ids)}): ").strip()
            inspector.inspect_galaxy(gal_id)
            
        elif choice == "2":
            gal_id = input(f"Enter galaxy ID (or 'all' for all galaxies): ").strip()
            if gal_id.lower() == 'all':
                inspector.plot_galaxy_footprint()
            else:
                inspector.plot_galaxy_footprint(gal_id)
                
        elif choice == "3":
            gal_id = input(f"Enter galaxy ID ({', '.join(inspector.galaxy_ids)}): ").strip()
            pixel_choice = input("Enter pixel index ('median' for median SED, 'max_snr' for max S/N pixel): ").strip()
            
            if pixel_choice.lower() == 'median':
                pixel_idx = None
            elif pixel_choice.lower() == 'max_snr': #averaged over filters F277W, F356W, F444W
                pixel_idx = 'max_snr'
            else:
                try:
                    pixel_idx = int(pixel_choice)
                except ValueError:
                    print("❌ Invalid pixel choice. Using median SED.")
                    pixel_idx = None
                    
            inspector.plot_sed(gal_id, pixel_idx)
            
        elif choice == "4":
            inspector.create_summary_table()
            
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

def main():
    """Main function to run the inspector"""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        local_path = os.path.dirname(os.path.abspath(__file__))
        filename = local_path + "/six_galaxies_data.hdf5"
    
    print("🌌 JADES Galaxy HDF5 Inspector")
    print("=" * 40)
    
    # Quick check mode vs interactive mode
    mode = input("Choose mode:\n1. Quick overview\n2. Interactive explorer\nEnter choice (1-2): ").strip()
    
    if mode == "1":
        # Quick overview mode
        try:
            inspector = JADESInspector(filename)
            inspector.print_overview()
            inspector.create_summary_table()
            
            # Show a sample galaxy
            sample_id = inspector.galaxy_ids[0]
            print(f"\n🌟 Sample Galaxy ({sample_id}):")
            inspector.inspect_galaxy(sample_id)
            
        except FileNotFoundError as e:
            print(f"❌ {e}")
    
    elif mode == "2":
        # Interactive mode
        interactive_explorer(filename)
    
    else:
        print("❌ Invalid choice.")

if __name__ == "__main__":
    main()