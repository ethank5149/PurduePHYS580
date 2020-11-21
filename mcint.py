from numpy import empty, mean, sin, cos, pi
from numpy.random import default_rng


def mcint(func, bounds, npoints=int(1e7)):
    rng = default_rng()
    low, high = bounds
    return mean(func(rng.uniform(size=npoints, low=low, high=high))) * (high - low)
    

def main():
    I = mcint(lambda x : cos(cos(x)) - cos(sin(x)), [0, 2 * pi])

    print('2π')
    print(f'∫cos(cos(x))-cos(sin(x))dx = 0 ~ {I:0.6g}')
    print('0')


if __name__ == '__main__':
    main()