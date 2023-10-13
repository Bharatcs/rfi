import healpy as hp

nside = 16
pixel_number = 1812

# Get the (x, y) coordinates within the pixel's hierarchical level
_, x, y = hp.pix2xy(nside, pixel_number)

quadrant = None

if x >= 0 and y >= 0:
    quadrant = 0
elif x < 0 and y >= 0:
    quadrant = 1
elif x < 0 and y < 0:
    quadrant = 2
elif x >= 0 and y < 0:
    quadrant = 3

print(f"HEALPix pixel number 1812 is in Quadrant {quadrant}.")

