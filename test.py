import numpy as np
l = np.full((2, 11), np.nan)
l[1, 5] = 4
l[0, 2] = 2
l[0, 7] = 3
non_nan_indices = np.argwhere(~np.isnan(l))
print(non_nan_indices)
print()
print(l)
print(l.flatten())
print(np.nanargmax(l))
print("Indices of maximum values (ignoring NaNs):", max_indices)