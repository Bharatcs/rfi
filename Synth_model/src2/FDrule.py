# Bin width using Freedman Diaconis rule
import numpy as np
def FDrule(data):
    # Calculate interquartile range
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    
    # Calculate Freedman-Diaconis bin width
    n = len(data)
    bin_width = 2 * iqr / (n ** (1/3))

    # Calculate number of bins
    data_range = max(data) - min(data)
    num_bins = int(data_range / bin_width) + 1
    
    return num_bins
