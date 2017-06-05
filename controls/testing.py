import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate

# finds the determinant of the covariance matrix
def cov_determinant(theta, sigma_x, sigma_y):
    return (-sigma_x**4)*theta**4 + (3*sigma_x**4)*theta**3 * (sigma_x**2*sigma_y**2 - 3*sigma_x**4)*theta**2 + \
                                    (sigma_x**4 - 2*sigma_x**2*sigma_y**2)*theta + (sigma_x**2*sigma_y**2)

# calculates the joint probability
def p_joint(x, y, theta, sigma_x, sigma_y):
    exp_term = -0.5 * (x**2*(theta*sigma_x**2 + sigma_y**2) + 2*x*y*(sigma_x**2*(theta**2-theta)) + y**2*(sigma_x**2*(1-theta)**2)) / cov_determinant(theta, sigma_x, sigma_y)
    ans = (2*np.pi)**-0.5 * cov_determinant(theta, sigma_x, sigma_y)**-0.5 * np.exp(exp_term)
    return ans

def p_x(x, theta, sigma_x):
    return (1 / np.sqrt(2*np.pi*sigma_x**2)) * np.exp(-(x - theta*x)**2 / (2*sigma_x**2))

def p_y(x, y, theta, sigma_y):
    return (1 / np.sqrt(2*np.pi*sigma_y**2)) * np.exp(-(theta*x + y) / (2*sigma_y**2))

def mutual_info(theta, sigma_x, sigma_y):
    def f(x,y):
        return p_joint(x,y,theta,sigma_x,sigma_y) * np.log(p_joint(x,y,theta,sigma_x,sigma_y) / (p_x(x,theta,sigma_x)*p_y(x,y,theta,sigma_y)))
    return integrate.dblquad(f, -2, 2, lambda x: -2, lambda x: 2)

data = []
for num in np.arange(0, 0.8, 0.05):
    data.append(mutual_info(num,1,1))
plt.plot(data)
plt.show()