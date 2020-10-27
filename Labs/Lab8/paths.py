from numpy import asarray, sin, cos, pi, vsplit


path_loop_top =      lambda s, r, l, n : asarray([                    r * cos(2 * pi * n * s),                     r * sin(2 * pi * n * s),           l / 2])
path_loop_bottom =   lambda s, r, l, n : asarray([                    r * cos(2 * pi * n * s),                     r * sin(2 * pi * n * s),         - l / 2])
path_rloop_bottom =  lambda s, r, l, n : asarray([              r * cos(2 * pi * n * (1 - s)),               r * sin(2 * pi * n * (1 - s)),         - l / 2])
path_solenoid =      lambda s, r, l, n : asarray([                    r * cos(2 * pi * n * s),                     r * sin(2 * pi * n * s), l * (s - 1 / 2)])
dpath_loop_top =     lambda s, r, l, n : asarray([      -2 * pi * n * r * sin(2 * pi * n * s),        2 * pi * n * r * cos(2 * pi * n * s),               0])
dpath_loop_bottom =  lambda s, r, l, n : asarray([      -2 * pi * n * r * sin(2 * pi * n * s),        2 * pi * n * r * cos(2 * pi * n * s),               0])
dpath_rloop_bottom = lambda s, r, l, n : asarray([ 2 * pi * n * r * sin(2 * pi * n * (1 - s)), -2 * pi * n * r * cos(2 * pi * n * (1 - s)),               0])
dpath_solenoid =     lambda s, r, l, n : asarray([      -2 * pi * n * r * sin(2 * pi * n * s),        2 * pi * n * r * cos(2 * pi * n * s),               l])