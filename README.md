**# SBIPIX**
Simulation-based inference of galaxy properties from JWST pixels
---
The spectral energy distributions (SEDs) of galaxies offer detailed insights into their stellar populations, capturing key physical properties such as stellar mass, star formation history (SFH), metallicity, and dust attenuation. However, inferring these properties from SEDs is a highly degenerate inverse problem, particularly when using integrated observations and a limited number of photometric bands. We present an efficient Bayesian SED-fitting framework tailored to multiwavelength pixel photometry from the JWST Advanced Deep Extragalactic Survey (JADES). Our method employs simulation-based inference to enable rapid posterior sampling across galaxy pixels, leveraging the unprecedented spatial resolution, wavelength coverage, and depth provided by the survey. It is trained on synthetic photometry generated from MILES stellar population models, incorporating both parametric and non-parametric SFHs, realistic noise, and JADES-like filter sensitivity thresholds.

## Getting Started

### Prerequisites

SBIPIX requires Python 3.8+


### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/patriglesias/SBIPIX.git
cd SBIPIX
```

2. **Install in development mode:**
```bash
pip install -e .
```

3. **Verify installation:**
```python
python -c "from sbipix import sbipix; print('SBIPIX installed successfully!')"
```

### Quick Start

#### 1. Initialize SBIPIX
```python
from sbipix import sbipix
from sbipix import *
from sbipix.plotting.diagnostics  import plot_test_performance

# Create SBIPIX instance
sx = sbipix()

# Configure for JADES
sx.filter_path = 'obs/obs_properties/'
sx.filter_list='filters_jades_no_wfc.dat'
sx.atlas_name = 'atlas_obs_jades'
sx.n_simulation = 1000
sx.parametric = True  # or False for non-parametric SFH
sx.both_masses = True
sx.infer_z = False
```

#### 2. Run a Basic Simulation
```python
# Simulate galaxy SEDs
sx.simulate(
    mass_max=12, mass_min=4.0,
    z_prior='flat', z_min=0.0, z_max=7.5,
    Z_min=-2.27, Z_max=0.4,
    dust_model='Calzetti',
    Av_min=0.0, Av_max=4.0
)

# Load simulation results
sx.load_simulation()
```

#### 3. Add Observational Effects
```python
# Load observational properties
sx.include_limit = True
sx.condition_sigma = True
sx.include_sigma = True
sx.load_obs_features()
sx.add_noise_nan_limit_all()
```

#### 4. Train the Model
```python
# Train normalizing flow
sx.train(
    min_thetas=min_values,
    max_thetas=max_values,
    n_max=len(sx.theta),
    nblocks=5,
    nhidden=128
)
```

#### 5. Test Performance
```python
# Test model performance
posterior_test = sx.test_performance(n_test=100, return_posterior=True)
sx.plot_test_performance(sx,n_test=100)

```

### Working with JADES Data

#### Inspect Sample Galaxies
```python
from examples.inspect_jades_hdf5 import JADESInspector

# Initialize inspector
inspector = JADESInspector("obs/six_galaxies_data.hdf5")

# Overview of the dataset
inspector.print_overview()

# Inspect specific galaxy
inspector.inspect_galaxy("205449")

# Plot galaxy footprint
inspector.plot_galaxy_footprint("205449")

# Plot SED (median, specific pixel, or max S/N pixel)
inspector.plot_sed("205449", pixel_idx='max_snr', show_upper_limits=True)
```

#### Run Inference on Sample Data
```python
from examples.inference_six_gal import galaxy_inference, run_inference_for_all_galaxies

# Single galaxy inference
posteriors = galaxy_inference(
    sx, 
    galaxy_id=205449,
    data="obs/six_galaxies_data.hdf5",
    sn_limit=5.0,
    n_samples=500,
    save_posteriors=True
)

# Process all sample galaxies
run_inference_for_all_galaxies(
    hdf5_file="obs/six_galaxies_data.hdf5",
    sn_limits=[5, 5, 5, 5, 5, 5]
)
```

#### Visualize Results
```python
from examples.galaxy_maps_sbipix import galaxy_maps, run_all_galaxies

# Create parameter maps for a single galaxy
galaxy_maps(
    sx=sx,
    galaxy_id=205449,
    size=75,
    plot='maps',  # 'phot', 'err', 'sn', or 'maps'
    cmap='turbo',
    plot_fraction=True
)

# Process all galaxies
run_all_galaxies(
    data_file="obs/six_galaxies_data.hdf5",
    plot_type='maps'
)
```

### Repository Structure

```
SBIPIX/
├── src/sbipix/           # Main package
│   ├── __init__.py       
│   ├── sbipix.py         # Core SBIPIX class
│   ├── plotting/         # Plotting utilities
│   │   └── diagnostics.py
│   ├── train/            # Training modules
│   │   └── simulator.py
│   └── utils/            # Utility functions
│       ├── cosmology.py
│       └── sed_utils.py
├── examples/             # Example scripts and tutorials
│   ├── inspect_jades_hdf5.py      # Data inspection tool
│   ├── inference_six_gal.py       # Inference pipeline
│   ├── galaxy_maps_sbipix.py      # Visualization tool
│ 
├── obs_properties/       # Observational data
│   ├── filters_jades_no_wfc.dat
│   └── background_noise_hainline.npy
├── setup.py
└── README.md
```

### Next Steps

- **Tutorial Notebooks**: See `examples/` for Jupyter notebooks with detailed walkthroughs
- **API Documentation**: Run `help(sbipix)` for full API reference  
- **Sample Data**: Download JADES sample data from [link]
- **Pre-trained Models**: Access trained models at [link]

### Citation

If you use SBIPIX in your research, please cite:
```bibtex
@misc{iglesiasnavarro2025simulationbasedinferencegalaxyproperties,
      title={Simulation-based inference of galaxy properties from JWST pixels}, 
      author={Patricia Iglesias-Navarro and Marc Huertas-Company and Pablo Pérez-González and Johan H. Knapen and ChangHoon Hahn and Anton M. Koekemoer and Steven L. Finkelstein and Natalia Villanueva and Andrés Asensio Ramos},
      year={2025},
      eprint={2506.04336},
      archivePrefix={arXiv},
      primaryClass={astro-ph.GA},
      url={https://arxiv.org/abs/2506.04336}, 
}
```