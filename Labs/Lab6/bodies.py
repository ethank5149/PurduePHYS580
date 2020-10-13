import numpy as np
from numpy import asarray, sin, cos

bodies = {
    'saturn' : {
        'pos' : [ 0.000000000000000E-00,  0.000000000000000E-00,  0.000000000000000E-00],
        'vel' : [ 0.000000000000000E-00,  0.000000000000000E-00,  0.000000000000000E-00],
        'm' : 1.0
    },
    "titan" : {
        'pos' : [-2.349442311566901E-03,  7.153850305740229E-03, -3.453819029042604E-03],
        'vel' : [-3.051170257091698E-03, -6.116575942904424E-04,  6.191676545191825E-04],
        'm' : 2.367e-4
    },
        'rhea' : {
        'pos' : [-1.255393996976257E-03,  2.964327117947999E-03, -1.436487271404112E-03],
        'vel' : [-4.561058618370766E-03, -1.378619573613746E-03,  1.133396533665187E-03],
        'm' : 4.062e-6
    },
    'hyperion' : {
        'pos' : [ 3.643573903413987E-03, -7.406358781088398E-03,  3.362810684983125E-03],
        'vel' : [ 2.818505976172301E-03,  1.278934691098328E-03, -9.114757023594637E-04],
        'm' : 9.754e-9,
        'r' : (9.024e-7, 1.0468e-6, 6.8651e-7),
        'T' : 21.28
    },
}


theta = np.pi
phi = np.pi/2.0
dtheta = 0.0*np.pi / bodies['hyperion']['T']
dphi = 0.0*np.pi / bodies['hyperion']['T']
r = max(bodies['hyperion']['r'])
bodies['hyperion_a'] = {
    'pos' : bodies['hyperion']['pos'] + r * asarray([sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)]), 
    'vel' : bodies['hyperion']['vel'] + r * asarray([dphi * cos(phi) * cos(theta) - dtheta * sin(phi) * sin(theta), dphi * cos(phi) * sin(theta) + dtheta * sin(phi) * cos(theta), -dphi * sin(phi)]), 
    'm' : 0.5*bodies['hyperion']['m']
}
bodies['hyperion_b'] = {
    'pos' : bodies['hyperion']['pos'] - r * asarray([sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)]), 
    'vel' : bodies['hyperion']['vel'] - r * asarray([dphi * cos(phi) * cos(theta) - dtheta * sin(phi) * sin(theta), dphi * cos(phi) * sin(theta) + dtheta * sin(phi) * cos(theta), -dphi * sin(phi)]), 
    'm' : 0.5*bodies['hyperion']['m']
}

