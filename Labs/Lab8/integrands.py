from numpy import asarray, sin, cos, pi

def B_loop_top_x_integrand(s, x, y, z, l, R, w):
    num = 2 * pi * R * w * (l / 2 + z) * cos(2 * pi * w * s)
    den = ((z + l / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den

def B_loop_top_y_integrand(s, x, y, z, l, R, w):
    num = 2 * pi * R * w * (l / 2 + z) * sin(2 * pi * w * s)
    den = ((z + l / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den

def B_loop_top_z_integrand(s, x, y, z, l, R, w):
    num = 2 * pi * R * w * (R - x * cos(2 * pi * w * s) - y * sin(2 * pi * w * s))
    den = ((z + l / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den

def B_loop_bottom_x_integrand(s, x, y, z, l, R, w):
    num =  -2 * pi * R * w * (l / 2 - z) * cos(2 * pi * w * s)
    den = ((z - l / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den
    
def B_loop_bottom_y_integrand(s, x, y, z, l, R, w):
    num = -2 * pi * R * w * (l / 2 - z) * sin(2 * pi * w * s)
    den = ((z - l / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den

def B_loop_bottom_z_integrand(s, x, y, z, l, R, w):
    num = 2 * pi * R * w * (R - x * cos(2 * pi * w * s) - y * sin(2 * pi * w * s))
    den = ((z - l / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den


def B_loop_bottom_reversed_x_integrand(s, x, y, z, l, R, w):
    num =  -2 * pi * R * w * (l / 2 - z) * cos(2 * pi * w * s)
    den = ((z - l / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den
    
def B_loop_bottom_reversed_y_integrand(s, x, y, z, l, R, w):
    num = -2 * pi * R * w * (l / 2 - z) * sin(2 * pi * w * s)
    den = ((z - l / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den

def B_loop_bottom_reversed_z_integrand(s, x, y, z, l, R, w):
    num = 2 * pi * R * w * (R - x * cos(2 * pi * w * s) - y * sin(2 * pi * w * s))
    den = ((z - l / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den


def B_solenoid_x_integrand(s, x, y, z, l, R, w):
    num = l * (R * sin(2 * pi * w * s) - y) + 2 * pi * R * w * (z - l * (2 * s - 1) / 2) * cos(2 * pi * w * s)
    den = ((z - l * (2 * s - 1) / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den

def B_solenoid_y_integrand(s, x, y, z, l, R, w):
    num = l * (x - R * cos(2 * pi * w * s)) - 2 * pi * R * w * (l * (2 * s - 1) / 2 - z) * sin(2 * pi * w * s)
    den = ((z - l * (2 * s - 1) / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den

def B_solenoid_z_integrand(s, x, y, z, l, R, w):
    num = 2 * pi * R * w * (R - x * cos(2 * pi * w * s) - y * sin(2 * pi * w * s))
    den = ((z - l * (2 * s - 1) / 2) ** 2 + (y - R * sin(2 * pi * w * s)) ** 2 + (x - R * cos(2 * pi * w * s)) ** 2) ** (3 / 2)
    return num / den