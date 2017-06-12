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

def generate_data(mu_x, mu_y, sigma_x, sigma_y, size=100000):
    x_data = []
    y_data = []
    for num in range(size):
        x_data.append(np.random.normal(loc=mu_x, scale=sigma_x))
        y_data.append(np.random.normal(loc=mu_y, scale=sigma_y))
    return np.vstack((x_data, y_data))

def mutual_info(sigma_x, sigma_y, theta, interval=0.1, lower_limit=-5, upper_limit=5):
    total = 0
    for x in np.arange(lower_limit, upper_limit, interval):
        for y in np.arange(lower_limit, upper_limit, interval):
            total += p_joint(x, y, theta, sigma_x, sigma_y) * np.log(p_joint(x, y, theta, sigma_x, sigma_y)
                                                                             / (p_x(x, sigma_x, theta)*p_y(y, sigma_x, sigma_y, theta)))
    total /= 10000
    return total

def estimate_mutual_info(data_matrix, resolution=50):
    x_data = data_matrix[0]
    y_data = data_matrix[1]
    x_kde = gaussian_kde(x_data)
    y_kde = gaussian_kde(y_data)
    xy_kde = gaussian_kde(data_matrix)
    def f(x,y):
        return xy_kde((x,y)) * np.log(xy_kde((x,y)) / (x_kde(x) * y_kde(y)))
    total = 0
    for x in np.arange(-5,5,10.0/resolution):
        for y in np.arange(-5,5,10.0/resolution):
            total += f(x,y)
    total /= resolution**2
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
    data = generate_data(0,0,sigma_x,sigma_y)
    results = []
    expected = []
    test_y_dist()
    for theta in np.arange(0,0.6,0.1):
        new_sigma_x = (1-theta) * sigma_x
        new_sigma_y = np.sqrt(sigma_y**2 + (sigma_x**2*theta**2))
        spill = np.array([[1-theta, 0],
                          [theta, 1]])
        new_data = np.dot(spill, data)
        results.append(np.corrcoef(new_data)[0][1])
        expected.append((theta-theta**2) * sigma_x / new_sigma_y / new_sigma_x)
    plt.plot(results)
    plt.plot(expected)
    '''
    true_theta = 0
    spillover = np.array([[1-true_theta, 0],
                          [true_theta, 1]])
    data = np.dot(spillover, data)

    results = []
    test_results = []
    for theta in np.arange(0.3,0.8,0.1):
        spillover = np.array([[1-theta, 0],
                              [theta, 1]])
        compensation = np.linalg.inv(spillover)
        new_data = np.dot(compensation, data)
        new_data[0] = new_data[0] - np.mean(new_data[0])
        new_data[1] = new_data[1] - np.mean(new_data[1])
        results.append(np.std(new_data[0]))
        print(spillover)
        results.append(mutual_info(theta,1,1))
        test_results.append(estimate_mutual_info(new_data))
        '''
    #plt.plot(np.arange(0,1,0.1), test_results)
    plt.show()
