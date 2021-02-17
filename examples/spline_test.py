from kpicdrp.utils import get_spline_model,get_piecewise_linear_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

if __name__ == "__main__":
    N_knots = 5
    x_knots = np.linspace(0,4,N_knots, endpoint=True)
    y_vec = np.random.randn(N_knots)

    x_samples = np.linspace(0,4,5*N_knots, endpoint=True)
    M_lin = get_piecewise_linear_model(x_knots,x_samples)
    y_lin = np.dot(M_lin, y_vec)

    M_spl = get_spline_model(x_knots,x_samples)
    y_spl = np.dot(M_spl, y_vec)

    spl = InterpolatedUnivariateSpline(x_knots, y_vec, k=3, ext=0)
    y_spl_ref = spl(x_samples)

    plt.plot(x_knots,y_vec,"o",label="original")
    plt.plot(x_samples,y_lin,"x",label="piecewise linear")
    plt.plot(x_samples,y_spl,".",label="Spline (JB)")
    plt.plot(x_samples,y_spl_ref,label="Spline (ref)")
    plt.legend()
    plt.show()



