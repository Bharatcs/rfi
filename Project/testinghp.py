from PIL import Image
import numpy as np
from matplotlib.image import pil_to_array
import healpy as hp
grayscale_pil_image = Image.open("/media/sf_Shared_Ubuntu/water_16k.png").convert("L") ##grayscaled the image##
image_array = pil_to_array(grayscale_pil_image) ##converted the image into an array

