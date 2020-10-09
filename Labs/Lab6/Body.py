import numpy as np

class Body:
    def __init__(self, mass, body, 
                 length_scale=1.0, # Converts AU to another unit by: new_unit = old_unit / length_scale
                 time_scale=1.0,   # Converts d to another unit by: new_unit = old_unit / time_scale
                 mass_scale=1.0    # Converts kg to another unit by: new_unit = old_unit / mass_scale
                ):
        self.vecs = body.vectors()
        self.pos = np.asarray([self.vecs['x'][0], self.vecs['y'][0], self.vecs['z'][0]]) / length_scale
        self.vel = np.asarray([self.vecs['vx'][0], self.vecs['vy'][0], self.vecs['vz'][0]]) * time_scale / length_scale
        self.mass = mass / mass_scale