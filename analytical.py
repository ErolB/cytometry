import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy.stats import gaussian_kde

# finds the determinant of the covariance matrix
def cov_determinant(theta, sigma_x, sigma_y):
    return (-sigma_x**4)*theta**4 + (3*sigma_x**4)*theta**3 * (sigma_x**2*sigma_y**2 - 3*sigma_x**4)*theta**2 + \
                                    (sigma_x**4 - 2*sigma_x**2*sigma_y**2)*theta + (sigma_x**2*sigma_y**2)

# calculates the joint probability
def p_joint(x, y, theta, sigma_x, sigma_y):
    sigma_x *= (1-theta)
    sigma_y *= (1-theta)/np.sqrt(1-2*theta)
    exp_term = -0.5 * (x**2*(theta*sigma_x**2 + sigma_y**2) + 2*x*y*(sigma_x**2*(theta**2-theta)) + y**2*(sigma_x**2*(1-theta)**2)) / cov_determinant(theta, sigma_x, sigma_y)
    ans = (2*np.pi)**-1 * cov_determinant(theta, sigma_x, sigma_y)**-0.5 * np.exp(exp_term)
    return ans

def p_x(x, theta, sigma_x):
    sigma_x *= 1-theta
    return (1 / np.sqrt(2*np.pi*sigma_x**2)) * np.exp(-x**2 / (2*sigma_x**2))

def p_y(x, y, theta, sigma_y):
    sigma_y *= (1 - theta) / np.sqrt(1 - 2*theta)
    return (1 / np.sqrt(2*np.pi*sigma_y**2)) * np.exp(-(y-theta*x)**2 / (2*sigma_y**2))

def mutual_info(theta, sigma_x, sigma_y):
    def f(x,y):
        return p_joint(x,y,theta,sigma_x,sigma_y) * np.log(p_joint(x,y,theta,sigma_x,sigma_y) / (p_x(x,theta,sigma_x)*p_y(x,y,theta,sigma_y)))
    return integrate.dblquad(f, -5, 5, lambda x: -5, lambda x: 5)

def generate_data(mu_x, mu_y, sigma_x, sigma_y, size=10000):
    x_data = []
    y_data = []
    for num in range(size):
        x_data.append(np.random.normal(loc=mu_x, scale=sigma_x))
        y_data.append(np.random.normal(loc=mu_y, scale=sigma_y))
    return np.vstack((x_data, y_data))

def estimate_mutual_info(data_matrix, resolution=50):
    x_data = data_matrix[0]
    y_data = data_matrix[1]
    x_kde = gaussian_kde(x_data)
    y_kde = gaussian_kde(y_data)
    xy_kde = gaussian_kde(np.vstack((x_data,y_data)))
    def f(x,y):
        return xy_kde((x,y)) * np.log(xy_kde((x,y)) / (x_kde(x) * y_kde(y)))
    total = 0
    for x in np.arange(-2,2,10.0/resolution):
        for y in np.arange(-2,2,2.0/resolution):
            total += f(x,y)
    total /= resolution**2
    return total

data = generate_data(0,0,1,1)
true_theta = 0
spillover = np.array([[1-true_theta, 0],
                      [true_theta, 1]])
data = np.dot(spillover, data)

results = []
for theta in np.arange(0,0.5,0.05):
    spillover = np.array([[1-theta, 0],
                          [theta, 1]])
    compensation = np.linalg.inv(spillover)
    new_data = np.dot(compensation, data)
    print(spillover)
    results.append(mutual_info(theta,1,1))
    

plt.plot(np.arange(0,0.5,0.05), results)
plt.show()
'''

data = generate_data(1,0,1,1)
print(data)
theta = 0.2
spillover = np.array([[1-theta, 0],
                      [theta, 1]])
data = np.dot(spillover, data)
print(data)
y_kde = gaussian_kde(data[1])
print(np.mean(data[1]))
print(np.std(data[1]))

y_values1 = []
y_values2 = []
for y in np.arange(-5,5,0.1):
    y_values1.append(p_y(1,y,theta,1))
    y_values2.append(y_kde(y))

plt.plot(np.arange(-5,5,0.1), y_values1)
plt.plot(np.arange(-5,5,0.1), y_values2)

plt.show()
'''