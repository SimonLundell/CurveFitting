import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

"""
Here is a model where beta=(a,b,c)  and

We want to find the optimal values of a, b and c that fit the data below (the shape is a parabola)

This requires two steps

1. Define model function
2. Use scipy's curve_fit function. This requires giving initial guesses for , 
which one needs to obtain carefuly for complicated models, or else the function might not work.
"""

x_data = np.array([0.        , 0.15789474, 0.31578947, 0.47368421, 0.63157895,
       0.78947368, 0.94736842, 1.10526316, 1.26315789, 1.42105263,
       1.57894737, 1.73684211, 1.89473684, 2.05263158, 2.21052632,
       2.36842105, 2.52631579, 2.68421053, 2.84210526, 3.        ])

y_data = np.array([  2.95258285,   2.49719803,  -2.1984975 ,  -4.88744346,
        -7.41326345,  -8.44574157, -10.01878504, -13.83743553,
       -12.91548145, -15.41149046, -14.93516299, -13.42514157,
       -14.12110495, -17.6412464 , -16.1275509 , -16.11533771,
       -15.66076021, -13.48938865, -11.33918701, -11.70467566])

def plot(fit_x, fit_y) -> None:
    plt.scatter(x_data, y_data)
    plt.plot(fit_x, fit_y, '--r')
    plt.show()
    return

def model_f(x,a,b,c) -> float:
    return a * (x-b)**2 + c

def main():
    p_optimized, p_covariance = curve_fit(model_f, x_data, y_data, p0=[3,2,-16])
    a_opt, b_opt, c_opt = p_optimized
    x_model = np.linspace(min(x_data), max(x_data), 100)
    y_model = model_f(x_model, a_opt, b_opt, c_opt)
    plot(x_model, y_model)
    return

if __name__ == '__main__':
    main()