"""
Cosmology utilities for SBIPIX.
"""

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


def setup_cosmology(H0=70, Om0=0.3):
    """
    Set up standard cosmology for SBIPIX.
    
    Parameters
    ----------
    H0 : float, optional
        Hubble constant in km/s/Mpc (default: 70)
    Om0 : float, optional
        Matter density parameter (default: 0.3)
        
    Returns
    -------
    astropy.cosmology.FlatLambdaCDM
        Cosmology object
    """
    return FlatLambdaCDM(H0=H0, Om0=Om0)


def angular_diameter_distance(z, cosmo=None):
    """
    Compute the angular diameter distance D_A(z) in kpc/arcsec.
    
    Parameters
    ----------
    z : float
        Redshift
    cosmo : astropy.cosmology object, optional
        Cosmology (default: creates FlatLambdaCDM with H0=70, Om0=0.3)
    
    Returns
    -------
    float
        Angular diameter distance in kpc/arcsec
    """
    if cosmo is None:
        cosmo = setup_cosmology()
        
    D_A = cosmo.angular_diameter_distance(z).to(u.kpc)  # Convert to kpc
    kpc_per_arcsec = D_A / 206265.0  # Convert to kpc/arcsec (206265 arcsec/radian)
    return kpc_per_arcsec.value  # Return as float


def age_at_redshift(z, cosmo=None):
    """
    Compute the age of the universe at redshift z.
    
    Parameters
    ----------
    z : float or np.ndarray
        Redshift(s)
    cosmo : astropy.cosmology object, optional
        Cosmology (default: creates FlatLambdaCDM with H0=70, Om0=0.3)
        
    Returns
    -------
    float or np.ndarray
        Age of universe in Gyr
    """
    if cosmo is None:
        cosmo = setup_cosmology()
        
    return cosmo.age(z).to(u.Gyr).value


def lookback_time(z, cosmo=None):
    """
    Compute lookback time to redshift z.
    
    Parameters
    ----------
    z : float or np.ndarray
        Redshift(s)
    cosmo : astropy.cosmology object, optional
        Cosmology (default: creates FlatLambdaCDM with H0=70, Om0=0.3)
        
    Returns
    -------
    float or np.ndarray
        Lookback time in Gyr
    """
    if cosmo is None:
        cosmo = setup_cosmology()
        
    return cosmo.lookback_time(z).to(u.Gyr).value


def comoving_distance(z, cosmo=None):
    """
    Compute comoving distance to redshift z.
    
    Parameters
    ----------
    z : float or np.ndarray
        Redshift(s)
    cosmo : astropy.cosmology object, optional
        Cosmology (default: creates FlatLambdaCDM with H0=70, Om0=0.3)
        
    Returns
    -------
    float or np.ndarray
        Comoving distance in Mpc
    """
    if cosmo is None:
        cosmo = setup_cosmology()
        
    return cosmo.comoving_distance(z).to(u.Mpc).value


def luminosity_distance(z, cosmo=None):
    """
    Compute luminosity distance to redshift z.
    
    Parameters
    ----------
    z : float or np.ndarray
        Redshift(s)
    cosmo : astropy.cosmology object, optional
        Cosmology (default: creates FlatLambdaCDM with H0=70, Om0=0.3)
        
    Returns
    -------
    float or np.ndarray
        Luminosity distance in Mpc
    """
    if cosmo is None:
        cosmo = setup_cosmology()
        
    return cosmo.luminosity_distance(z).to(u.Mpc).value