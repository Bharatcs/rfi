import numpy as np
import imageio.v2 as imageio
from astropy.io import fits

def tif_to_fits(tif_file, fits_file):
    # Read the TIFF image using imageio
    tif_array = imageio.imread(tif_file)

    # Create a FITS HDU (Header/Data Unit)
    hdu = fits.PrimaryHDU(tif_array)

    # Write the FITS HDU to a file
    hdu.writeto(fits_file, overwrite=True)

if __name__ == "__main__":
    # Replace 'input.tif' with the path to your .tif file
    input_tif_file = "/home/pratush/Downloads/GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0/GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0.tif"

    # Replace 'output.fits' with the desired path for your .fits file
    output_fits_file = "/home/pratush/Downloads/tiffitsconvert.fits"

    tif_to_fits(input_tif_file, output_fits_file)

