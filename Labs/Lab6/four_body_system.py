import numpy as np

def four_body_system(t, y, mu, G):
    m1, m2, m3, m4 = mu
    x1, x2, x3, x4, v1, v2, v3, v4 = np.hsplit(y, 8)
    return np.hstack((
        v1,
        v2,
        v3,
        v4,
        -G*(m2*(x1-x2)/np.linalg.norm(x1-x2)**3+m3*(x1-x3)/np.linalg.norm(x1-x3)**3+m4*(x1-x4)/np.linalg.norm(x1-x4)**3),
        -G*(m1*(x2-x1)/np.linalg.norm(x2-x1)**3+m3*(x2-x3)/np.linalg.norm(x1-x3)**3+m4*(x1-x4)/np.linalg.norm(x2-x4)**3),
        -G*(m1*(x3-x1)/np.linalg.norm(x3-x1)**3+m2*(x3-x2)/np.linalg.norm(x2-x2)**3+m4*(x3-x4)/np.linalg.norm(x3-x4)**3),
        -G*(m1*(x4-x1)/np.linalg.norm(x4-x1)**3+m2*(x4-x2)/np.linalg.norm(x4-x2)**3+m3*(x4-x3)/np.linalg.norm(x4-x3)**3)
    ))