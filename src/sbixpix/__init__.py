"""
SBIPIX: Simulation-Based Inference for Pixel-level Stellar Population Analysis
"""

__version__ = "0.1.0"
__author__ = "Patricia Iglesias-Navarro"
__email__ = "patriglesiasnavarro@gmail.com"
__uri__ = "https://github.com/patriglesias/sbipix"
__license__ = "MIT"
__description__ = "Simulation-based inference for pixel-level stellar population properties from galaxy SEDs"

# Import main class
from .sbipix import SBIPIX

# Import key utilities
from .utils.sed_utils import *
from .utils.cosmology import *

# Import plotting functions
from .plotting.diagnostics  import *