# Minimum value in the dataset is = -32768, Maximum value in the dataset is = 32767
# This function rescales all values to be between -1 and 1.

def normalize(x):
    result = 2 * ((x - (-32768)) / (32767 - (-32768))) - 1
    return result