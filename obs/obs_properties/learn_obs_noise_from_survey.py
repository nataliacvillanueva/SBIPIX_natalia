# Extract the observational distribution of uncertainties for the HUDF using HST (ACS) and JADES mosaics (before PSF matching)

import numpy as np
import os
from matplotlib import pyplot as plt
from astropy.io import fits
from sbipix.utils.sed_utils import mag_conversion


def get_filenames_in_dir(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
 
#download your mosaics here! they should be properly aligned and with the same pixel scale

#i.e
"""
from astroquery.mast import Observations
f115w_obs = Observations.query_criteria(provenance_name="jades",instrument_name='NIRCAM/IMAGE',filters='F115W')
data_products = Observations.get_product_list(f115w_obs)
Observations.download_products(data_products)"""

#download also the segmentation map
"""from astroquery.mast import Observations
segmentation = Observations.query_criteria(provenance_name="jades",instrument_name='NIRCAM/IMAGE',filters='segmentation')
data_products = Observations.get_product_list(segmentation)
#Observations.download_products(data_products)"""

#local path to the segmentation map and mosaics
path='/Users/patriglesias/Desktop/PhD/sedflow_JWST/obs'

#load the segmentation map
seg = fits.open(path+'/hlsp_jades_jwst_nircam_goods-s-deep_segmentation_v2.0_drz.fits')[0]

#pixels selected as galaxies in the JADES segmentation map, in my region of interest
xx=seg.data[-15000:, :16400][5500:10000, 5500:10000]!=0 #boolean mask of the pixels selected as galaxies in the JADES segmentation map

#load the mosaics names
nircam_mosaics=sorted(get_filenames_in_dir(path+'/nircam_dr2/'))[1:] #skip .DS_Store file
acs_mosaics=sorted(get_filenames_in_dir(path+'/hudf_aligned/'))
acs_err_mosaics=sorted(get_filenames_in_dir(path+'/hudf_aligned_err/'))

#save the photometry and errors in the galaxies of the JADES mosaics, in my region of interest
phot_jades=[]
err_jades=[]
for k,j in enumerate(nircam_mosaics[:]):
    print(j)
    hdul = fits.open(path+'/nircam_dr2/'+j)
    #load the photometry and errors in my region of interest
    phot=hdul[1].data[-15000:, :16400][5500:10000, 5500:10000]
    err=hdul[2].data[-15000:, :16400][5500:10000, 5500:10000]
    #convert to microJy
    PIXAR_SR= hdul[1].header['PIXAR_SR']
    phot=phot*PIXAR_SR*1e12 #microJy
    err=err*PIXAR_SR*1e12 #microJy
    phot_jades.append(phot[xx])
    err_jades.append(err[xx])
    hdul.close()

#same for the ACS mosaics
#Now with ACS
phot_acs=[]
err_acs=[]
for k,j in enumerate(acs_mosaics[:]):
    print(j)
    hdul = fits.open(path+'/hudf_aligned/'+j)
    photo  = hdul[0].data[-15000:, :16400][5500:10000, 5500:10000]
    #change units e-/s to microJansky
    #use https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints 
    factor_flux=hdul[0].header['PHOTFLAM']
    photo = photo * factor_flux  #from e-/s to erg/sec/cm^2/Angstrom
    lambda_filter=hdul[0].header['PHOTPLAM'] #in Angstrom
    photo = photo * 1e29 * lambda_filter**2  * 1e-8 / (2.998e10)#from  erg/(sec*cm^2*Angstrom) to microJy
    
    hdul = fits.open(path+'/hudf_aligned_err/'+acs_err_mosaics[k])
    err=1/np.sqrt(hdul[0].data)[-15000:, :16400][5500:10000, 5500:10000]
    #change units e-/s to microJansky
    #use https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints 
    err = err * factor_flux  #from e-/s to erg/sec/cm^2/Angstrom
    err= err * 1e29 * lambda_filter**2  * 1e-8 / (2.998e10) #from  erg/(sec*cm^2*Angstrom) to microJy
    phot_acs.append(photo[xx])
    err_acs.append(err[xx])
    hdul.close()

phot=np.concatenate((phot_jades,phot_acs),axis=0)
err=np.concatenate((err_jades,err_acs),axis=0)
print(np.shape(err)) #n_filters, n_pixels

mag=mag_conversion(phot,convert_to='mag')
mag_err=err*np.abs(-2.5/(np.log(10)*phot))

#divide the magnitudes into bins using percentiles (3 percentiles dividing the data into 4 bins)
percentiles = np.nanpercentile(mag, [10, 30, 40],axis=1)
print(percentiles)

index_percentiles=np.zeros((19,4),dtype=object) 
for i in range(19):
    index_percentiles[i,0]=np.where((mag[i,:]<percentiles[0,i]))[0]
    index_percentiles[i,1]=np.where((mag[i,:]>percentiles[0,i])&(mag[i,:]<percentiles[1,i]))[0]
    index_percentiles[i,2]=np.where((mag[i,:]>percentiles[1,i])&(mag[i,:]<percentiles[2,i]))[0]
    index_percentiles[i,3]=np.where((mag[i,:]>percentiles[2,i]))[0]

#calculate the mean and std of the errors of the magnitudes for the pixels in each bin for each filter
mag_err_ok=np.zeros((19,4),dtype=object)
mean_sigma=np.zeros((19,4))  
std_sigma=np.zeros((19,4))
for i in range(19):
   for j in range(4):
        try:
            mag_err_ok[i,j]=mag_err[i,index_percentiles[i,j]]
            mean_sigma[i,j]=np.nanmean(mag_err_ok[i,j])
            std_sigma[i,j]=np.nanstd(mag_err_ok[i,j])
        except:
            mag_err_ok[i,j]=np.array([])
            mean_sigma[i,j]=np.nan
            std_sigma[i,j]=np.nan

#visualize!
for i in range(19): #iterate over filters
    for j in range(3): #iterate over bins
        try:
            gaussian = np.random.normal(mean_sigma[i,j], std_sigma[i,j], 1000)
            plt.hist(gaussian, bins=10, alpha=0.5, label=f'Bin {j+1} - Filter {i+1}', density=True)
        except:
            print(f'No data for Bin {j+1} - Filter {i+1}')
            continue
    plt.xlabel('$\sigma$ [mag]')
    plt.ylabel('Distribution of noise')
    plt.legend()
    plt.show()

#store the mean and std of the errors of the magnitudes for the pixels in each bin for each filter
#np.save('mean_sigma_jades_res_bins.npy', mean_sigma)
#np.save('std_sigma_jades_res_bins.npy',std_sigma)
#store the percentiles for dividing in bins
#np.save('percentiles_jades_res_bins.npy', percentiles)


