import healpy as hp
import numpy as np
def shapecheck(lon,lat,name):
    checkpix=hp.ang2pix(128,lon,lat,lonlat=True)
    checkmap=np.zeros(hp.nside2npix(128))
    for pi in checkpix:
        checkmap[pi]+=1
    checkmap=np.where(checkmap != 0, 1, 0)
    return hp.mollview(checkmap,flip='geo',title=name,cmap='inferno')
    

