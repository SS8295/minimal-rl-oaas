#from customoaas import *
import numpy as np
import matplotlib.pyplot as plt

floorplan = np.array([[1,1,1,1,1,1,1,1,1,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [1,0,0,0,0,1,1,1,0,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [1,0,1,1,1,1,1,0,0,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [1,0,1,1,1,1,0,0,0,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [1,1,1,1,1,1,1,1,1,1]])
cmap = plt.cm.jet
norm = plt.Normalize(vmin=floorplan.min(), vmax=floorplan.max())
image = cmap(norm(floorplan))
plt.imsave('helper_functions/test.png', floorplan)