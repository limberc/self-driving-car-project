import matplotlib.pyplot as plt
import numpy as np

'''
All the pol in this file is the input of an numpy array
'''

xm_in_px = 3.675 / 85  # Lane width (12 ft in m) is ~85 px on image
ym_in_px = 3.048 / 24  # Dashed line length (10 ft in m) is ~24 px on image
MAX_RADIUS = 10000  # Calculate radius of curvature of a line
EQUID_POINTS = 25  # Number of points to use for the equidistant approximation


# Calculate coefficients of a polynomial in y+h coordinates, i.e. f(y) -> f(y+h)
def pol_shift(pol, h):
    pol_ord = len(pol) - 1  # Determinate degree of the polynomial
    if pol_ord == 3:
        pol0 = pol[0]
        pol1 = pol[1] + 3.0 * pol[0] * h
        pol2 = pol[2] + 3.0 * pol[0] * h * h + 2.0 * pol[1] * h
        pol3 = pol[3] + pol[0] * h * h * h + pol[1] * h * h + pol[2] * h
        return (np.array([pol0, pol1, pol2, pol3]))
    if pol_ord == 2:
        pol0 = pol[0]
        pol1 = pol[1] + 2.0 * pol[0] * h
        pol2 = pol[2] + pol[0] * h * h + pol[1] * h
        return (np.array([pol0, pol1, pol2]))
    if pol_ord == 1:
        pol0 = pol[0]
        pol1 = pol[1] + pol[0] * h
        return (np.array([pol0, pol1]))


# Calculate derivative for a polynomial pol in a point x
def pol_d(pol, x):
    pol_ord = len(pol) - 1
    if pol_ord == 3:
        return 3.0 * pol[0] * x * x + 2.0 * pol[1] * x + pol[2]
    if pol_ord == 2:
        return 2.0 * pol[0] * x + pol[1]
    if pol_ord == 1:
        # return pol[0] * np.ones(len(np.array(x)))
        return pol[0]


# Calculate the second derivative for a polynomial pol in a point x
def pol_dd(pol, x):
    pol_ord = len(pol) - 1
    if pol_ord == 3:
        return 6.0 * pol[0] * x + 2.0 * pol[1]
    if pol_ord == 2:
        return 2.0 * pol[0]
    if pol_ord == 1:
        return 0.0


# Calculate a polinomial value in a given point x
def pol_calc(pol, x):
    pol_f = np.poly1d(pol)
    return (pol_f(x))


def px_to_m(px):  # Conver ofset in pixels in x axis into m
    return xm_in_px * px


# Calculate offset from the lane center
def lane_offset(left, right):
    offset = 1280 / 2.0 - (pol_calc(left, 1.0) + pol_calc(right, 1.0)) / 2.0
    return px_to_m(offset)


def r_curv(pol, y):
    if len(pol) == 2:  # If the polinomial is a linear function
        return MAX_RADIUS
    else:
        y_pol = np.linspace(0, 1, num=EQUID_POINTS)
        x_pol = pol_calc(pol, y_pol) * xm_in_px
        y_pol = y_pol * 223 * ym_in_px
        pol = np.polyfit(y_pol, x_pol, len(pol) - 1)
        d_y = pol_d(pol, y)
        dd_y = pol_dd(pol, y)
        r = ((np.sqrt(1 + d_y ** 2)) ** 3) / abs(dd_y)
        if r > MAX_RADIUS:
            r = MAX_RADIUS
        return r


# Calculate radius of curvature of a lane by avaraging lines curvatures
def lane_curv(left, right):
    l = r_curv(left, 1.0)
    r = r_curv(right, 1.0)
    if l < MAX_RADIUS and r < MAX_RADIUS:
        return (r_curv(left, 1.0) + r_curv(right, 1.0)) / 2.0
    else:
        if l < MAX_RADIUS:
            return l
        if r < MAX_RADIUS:
            return r
        return MAX_RADIUS


# Calculate approximated equidistant to a parabola
def equidistant(pol, d, max_l=1, plot=False):
    y_pol = np.linspace(0, max_l, num=EQUID_POINTS)
    x_pol = pol_calc(pol, y_pol)
    y_pol *= 223  # Convert y coordinates to [0..223] scale
    x_m = []
    y_m = []
    k_m = []
    for i in range(len(x_pol) - 1):
        x_m.append((x_pol[i + 1] - x_pol[i]) / 2.0 + x_pol[i])  # Calculate polints position between given points
        y_m.append((y_pol[i + 1] - y_pol[i]) / 2.0 + y_pol[i])
        if x_pol[i + 1] == x_pol[i]:
            k_m.append(1e8)  # A vary big number
        else:
            k_m.append(-(y_pol[i + 1] - y_pol[i]) / (x_pol[i + 1] - x_pol[i]))  # Slope of perpendicular lines
    x_m = np.array(x_m)
    y_m = np.array(y_m)
    k_m = np.array(k_m)
    # Calculate equidistant points
    y_eq = d * np.sqrt(1.0 / (1 + k_m ** 2))
    x_eq = np.zeros_like(y_eq)
    if d >= 0:
        for i in range(len(x_m)):
            if k_m[i] < 0:
                y_eq[i] = y_m[i] - abs(y_eq[i])
            else:
                y_eq[i] = y_m[i] + abs(y_eq[i])
            x_eq[i] = (x_m[i] - k_m[i] * y_m[i]) + k_m[i] * y_eq[i]
    else:
        for i in range(len(x_m)):
            if k_m[i] < 0:
                y_eq[i] = y_m[i] + abs(y_eq[i])
            else:
                y_eq[i] = y_m[i] - abs(y_eq[i])
            x_eq[i] = (x_m[i] - k_m[i] * y_m[i]) + k_m[i] * y_eq[i]
    y_eq /= 223  # Convert all y coordinates back to [0..1] scale
    y_pol /= 223
    y_m /= 223
    pol_eq = np.polyfit(y_eq, x_eq, len(pol) - 1)  # Fit equidistant with a polinomial
    if plot:  # Visualize results
        plt.plot(x_pol, y_pol, color='red', linewidth=1, label='Original line')  # Original line
        plt.plot(x_eq, y_eq, color='green', linewidth=1, label='Equidistant')  # Equidistant
        plt.plot(pol_calc(pol_eq, y_pol), y_pol, color='blue',
                 linewidth=1, label='Approximation')  # Approximation
        plt.legend()
        for i in range(len(x_m)):
            plt.plot([x_m[i], x_eq[i]], [y_m[i], y_eq[i]], color='black', linewidth=1)  # Draw connection lines
        plt.savefig('output_images/equid.jpg')
    return pol_eq
