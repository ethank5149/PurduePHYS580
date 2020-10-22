from numpy import asarray, sin, cos, pi, vsplit

def path_loop_top(s, _l, _R, _w):
    x = asarray([_R * cos(2 * pi * _w * _) for _ in s])
    y = asarray([_R * sin(2 * pi * _w * _) for _ in s])
    z = asarray([                   _l / 2 for _ in s])
    return x, y, z

def path_loop_bottom(s, _l, _R, _w):
    x = asarray([_R * cos(2 * pi * _w * _) for _ in s])
    y = asarray([_R * sin(2 * pi * _w * _) for _ in s])
    z = asarray([                  -_l / 2 for _ in s])
    return x, y, z

def path_loop_bottom_reversed(s, _l, _R, _w):
    x = asarray([_R * cos(2 * pi * _w * (1 - _)) for _ in s])
    y = asarray([_R * sin(2 * pi * _w * (1 - _)) for _ in s])
    z = asarray([                        -_l / 2 for _ in s])
    return x, y, z

def path_solenoid(s, _l, _R, _w):
    x = asarray([_R * cos(2 * pi * _w * _) for _ in s])
    y = asarray([_R * sin(2 * pi * _w * _) for _ in s])
    z = asarray([     _l * (2 * _ - 1) / 2 for _ in s])
    return x, y, z