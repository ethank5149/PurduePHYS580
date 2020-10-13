import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.time import Time

bodies = {
    'sun'      : {'id' :   '10', 'id_type' : 'majorbody', 'm' : 1.9891e30,  'a' : 0.0,       'p' : 0.0},
    'mercury'  : {'id' :  '199', 'id_type' : 'majorbody', 'm' : 3.3011e23,  'a' : 0.0,       'p' : 0.0},
    'venus'    : {'id' :  '299', 'id_type' : 'majorbody', 'm' : 4.8675e24,  'a' : 0.0,       'p' : 0.0},
    'earth'    : {'id' :  '399', 'id_type' : 'majorbody', 'm' : 5.97e24,    'a' : 0.0,       'p' : 0.0},
    'mars'     : {'id' :  '499', 'id_type' : 'majorbody', 'm' : 6.417e23,   'a' : 0.0,       'p' : 0.0},
    'jupiter'  : {'id' :  '599', 'id_type' : 'majorbody', 'm' : 1.8982e27,  'a' : 0.0,       'p' : 0.0},
    'saturn'   : {'id' :  '699', 'id_type' : 'majorbody', 'm' : 5.6834e26,  'a' : 0.0,       'p' : 10759.22},
    'uranus'   : {'id' :  '799', 'id_type' : 'majorbody', 'm' : 8.6810e25,  'a' : 0.0,       'p' : 0.0},
    'neptune'  : {'id' :  '899', 'id_type' : 'majorbody', 'm' : 1.02413e26, 'a' : 0.0,       'p' : 0.0},
    'pluto'    : {'id' :  '999', 'id_type' : 'majorbody', 'm' : 1.303e22,   'a' : 0.0,       'p' : 0.0},
    'titan'    : {'id' :  '606', 'id_type' : 'smallbody', 'm' : 1.3455e23,  'a' : 1221870.0, 'p' : 15.945},
    'rhea'     : {'id' :  '605', 'id_type' : 'smallbody', 'm' : 2.31e21,    'a' : 527108.0,  'p' : 4.518212},    
    'hyperion' : {'id' :  '607', 'id_type' : 'smallbody', 'm' : 1.08e19,    'a' : 1481009.0, 'p' : 21.276},
    'iapetus'  : {'id' :  '608', 'id_type' : 'smallbody', 'm' : 1.6e21,     'a' : 0.0,       'p' : 0.0},
    'moon'     : {'id' : 'luna', 'id_type' : 'smallbody', 'm' : 7.342e22,   'a' : 0.0,       'p' : 0.0}}


default_observer = "699"
backup_observer = "10"

class Object:                   # define the objects: the Sun, Earth, Mercury, etc
    def __init__(self, name, color, ax, start_date, adjust=0.0):
        self.name = name
        observer = default_observer if name is not default_observer else backup_observer

        obj = Horizons(
            id=bodies[name.lower()]['id'], 
            id_type=bodies[name.lower()]['id_type'], 
            location=observer, 
            epochs=Time(start_date).jd)

        vec = obj.vectors()
        eph = obj.ephemerides()
        
        if self.name is not default_observer:
            self.r = np.asarray([vec['x'][0], vec['y'][0], vec['z'][0]])
            self.v = np.asarray([vec['vx'][0], vec['vy'][0], vec['vz'][0]]) 
        else:
            self.r = np.asarray([0.0, 0.0, 0.0])
            self.v = np.asarray([0.0, 0.0, 0.0]) 


        self.size = eph['ang_width']#1/(1 + np.exp(-x))
        self.m = bodies[name.lower()]['m']  

        self.xs = []
        self.ys = []
        ax.text(0, - (adjust + 0.1), name, color=color, zorder=1000, ha='center', fontsize='large')
        self.plot = ax.scatter(self.r[0], self.r[1], color=color, s=self.size**2, edgecolors=None, zorder=10)
        self.line, = ax.plot([], [], color=color, linewidth=1.4)