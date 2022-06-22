import numpy as np
import yaml

floorplan = np.array([[1,1,1,1,1,1,1,1,1,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [1,0,0,0,0,1,1,1,1,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [1,0,1,1,1,1,1,0,0,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [1,0,1,1,1,1,0,0,0,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [1,1,1,1,1,1,1,1,1,1]])

with open('config.yaml', 'w') as f:
    yaml.dump(floorplan.tolist(), f)

with open('config.yaml') as f:
    loaded = yaml.safe_load(f)

print(loaded)