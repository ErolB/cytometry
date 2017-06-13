import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy.stats import gaussian_kde

# finds the determinant of the covariance matrix
def cov_determinant(theta, sigma_x, sigma_y):
    return (-sigma_x**4)*theta**4 + (3*sigma_x**4)*theta**3 * (sigma_x**2*sigma_y**2 - 3*sigma_x**4)*theta**2 + \
                                    (sigma_x**4 - 2*sigma_x**2*sigma_y**2)*theta + (sigma_x**2*sigma_y**2)

def recalculate_sigmas(sigma_x, sigma_y, theta):
    new_sigma_x = (1 - theta) * sigma_x
    new_sigma_y = np.sqrt(sigma_y**2 + theta**2*sigma_x**2)
    return new_sigma_x, new_sigma_y

def correlation(sigma_x, sigma_y, theta):
    new_sigma_x, new_sigma_y = recalculate_sigmas(sigma_x, sigma_y, theta)
    return (theta-theta**2) * sigma_x**2 / (new_sigma_x * new_sigma_y)

# calculates the joint probability
def p_joint(x, y, theta, sigma_x, sigma_y):
    rho = correlation(sigma_x, sigma_y, theta)
    sigma_x, sigma_y = recalculate_sigmas(sigma_x, sigma_y, theta)
    z = (x**2 / sigma_x**2) - ((2*rho*x*y) / (sigma_x*sigma_y)) + (y**2 / sigma_y**2)
    exp_term = -(z / (2*(1-rho**2)))
    ans = (1 / (2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2))) * np.exp(exp_term)
    return ans

def p_x(x, sigma_x, theta):
    sigma_x, sigma_y = recalculate_sigmas(sigma_x, 1, theta)
    return (1 / np.sqrt(2*np.pi*sigma_x**2)) * np.exp(-x**2 / (2*sigma_x**2))

def p_y(y, sigma_x, sigma_y, theta):
    sigma_x, sigma_y = recalculate_sigmas(sigma_x, sigma_y, theta)
    return (1 / np.sqrt(2*np.pi*sigma_y**2)) * np.exp(-y**2 / (2*sigma_y**2))

'''
def mutual_info(theta, sigma_x, sigma_y):
    def f(x,y):
        return p_joint(x,y,theta,sigma_x,sigma_y) * np.log(p_joint(x,y,theta,sigma_x,sigma_y) / (p_x(x,theta,sigma_x)*p_y(x,y,theta,sigma_x,sigma_y)))
    return integrate.dblquad(f, -5, 5, lambda x: -5, lambda x: 5)[0]
'''

def generate_data(mu_x, mu_y, sigma_x, sigma_y, size=10000):
    x_data = []
    y_data = []
    for num in range(size):
        x_data.append(np.random.normal(loc=mu_x, scale=sigma_x))
        y_data.append(np.random.normal(loc=mu_y, scale=sigma_y))
    return np.vstack((x_data, y_data))

def mutual_info(sigma_x, sigma_y, theta, resolution = 50):

    new_sigma_x, new_sigma_y = recalculate_sigmas(sigma_x, sigma_y, theta)
    '''
    lower_x_lim = -3 * new_sigma_x
    upper_x_lim = 3 * new_sigma_x
    lower_y_lim = -3 * new_sigma_y
    upper_y_lim = 3 * new_sigma_y
    x_interval = (upper_x_lim - lower_x_lim) / resolution
    y_interval = (upper_y_lim - lower_y_lim) / resolution
    total = 0
    for x in np.arange(lower_x_lim, upper_x_lim, x_interval):
        for y in np.arange(lower_y_lim, upper_y_lim, y_interval):
            total += p_joint(x, y, theta, sigma_x, sigma_y) * np.log(p_joint(x, y, theta, sigma_x, sigma_y)
                                                                             / (p_x(x, sigma_x, theta)*p_y(y, sigma_x, sigma_y, theta)))
    total /= resolution**2
    return total
    '''
    rho = (theta-theta**2)*sigma_x**2 / (new_sigma_x*new_sigma_y)
    return (-1/2) * np.log10(1-rho**2) /100

def estimate_mutual_info(data_matrix, resolution=50):
    x_data = data_matrix[0]
    y_data = data_matrix[1]
    x_data -= np.mean(x_data)
    y_data -= np.mean(y_data)
    x_min = -10 * np.std(x_data)
    x_max = 10 * np.std(x_data)
    y_min = -10 * np.std(y_data)
    y_max = 10 * np.std(y_data)
    '''
    #x_min = -10
    #x_max = 10
    #y_min = -10
    #y_max = 10
    '''
    x_interval = float(x_max - x_min) / resolution
    y_interval = float(y_max - y_min) / resolution
    x_kde = gaussian_kde(x_data)
    y_kde = gaussian_kde(y_data)
    xy_kde = gaussian_kde(data_matrix)
    def f(x,y):
        ans = xy_kde((x,y)) * np.log(xy_kde((x,y)) / (x_kde(x) * y_kde(y)))
        if np.isnan(ans) or np.isinf(ans):
            return 0
        else:
            return ans
    total = 0
    for x in np.arange(x_min,x_max,x_interval):
        for y in np.arange(y_min,y_max,y_interval):
            total += f(x,y)
    total /= (resolution**2)
    return total

def test_y_dist():
    data = generate_data(0,0,1,1)
    print(data)
    theta = 0.5
    spillover = np.array([[1-theta, 0],
                          [theta, 1]])
    data = np.dot(spillover, data)
    print(data)
    kde = gaussian_kde(data[1])

    y_values1 = []
    y_values2 = []
    for y in np.arange(-5,5,0.1):
        y_values1.append(p_y(y,1))
        y_values2.append(kde(y))

    plt.plot(np.arange(-5,5,0.1), y_values1)
    plt.plot(np.arange(-5,5,0.1), y_values2)

    plt.show()

if __name__ == '__main__':
    sigma_x = 1
    sigma_y = 1
    data = generate_data(1000,10,sigma_x,sigma_y)

    true_theta = 0.25
    spillover = np.array([[1-true_theta, 0],
                          [true_theta, 1]])
    data = np.dot(spillover, data)
    #data = np.flipud(data)

    results = []
    test_results = []
    for theta in np.arange(0,1,0.1):
        spillover = np.array([[1-theta, 0],
                              [theta, 1]])
        compensation = np.linalg.inv(spillover)
        new_data = np.dot(compensation, data)
        print(spillover)
        test_results.append(estimate_mutual_info(new_data))

    plt.plot(test_results)
    plt.show()
