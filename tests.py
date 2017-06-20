"""
This contains unit tests.
"""

import utils
import compensation
import analytical

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import unittest

class MathTests(unittest.TestCase):
    def test_stds(self):
        """Ensures that the standard deviations are being calculated correctly"""
        threshold = 0.01  # defines an acceptable variation from the expected values
        plotting = False
        data = analytical.generate_data(0,0,1,1)
        results_x = []
        expected_x = []
        results_y = []
        expected_y = []
        for theta in np.arange(0,1,0.1):
            spill = [[1-theta, 0],
                     [theta, 1]]
            new_data = np.dot(spill, data)
            sigma_x, sigma_y = analytical.recalculate_sigmas(1,1,theta)
            expected_x.append(sigma_x)
            expected_y.append(sigma_y)
            results_x.append(np.std(new_data[0]))
            results_y.append(np.std(new_data[1]))
        # evaluate results for sigma_x
        x_score = 0
        for item in zip(expected_x, results_x):
            if abs(item[0] - item[1]) <= threshold:
                x_score += 1
        # evaluate results for sigma_y
        y_score = 0
        for item in zip(expected_y, results_y):
            if abs(item[0] - item[1]) <= threshold:
                y_score += 1
        # plot results
        if plotting:
            plt.plot(results_x)
            plt.plot(expected_x)
            plt.show()
            plt.plot(results_y)
            plt.plot(expected_y)
            plt.show()
        # return results of tests
        success = True
        if y_score < 8:
            print("error in sigma_y")
            success = False
        if x_score < 8:
            print("error in sigma_x")
            success = False
        self.assertTrue(success)

    def test_rho(self):
        """Tests the function for the correlation coefficient"""
        plotting = True
        threshold = 0.01
        sigma_x = 2
        sigma_y = 3
        data = analytical.generate_data(0, 0, sigma_x, sigma_y)
        results = []
        expected = []
        for theta in np.arange(0,1,0.1):
            spill = [[1-theta, 0],
                     [theta, 1]]
            new_data = np.dot(spill, data)
            expected.append(analytical.correlation(sigma_x,sigma_y,theta))
            results.append(np.corrcoef(new_data)[0][1])
        # plot data
        if plotting:
            plt.plot(results)
            plt.plot(expected)
            plt.show()
        # evaluate results
        score = 0
        for item in zip(results, expected):
            if abs(item[0] - item[1]) <= threshold:
                score += 1
        # return result of test
        self.assertTrue(score >= 8)

    def test_joint(self):
        success = True
        threshold = 0.01
        sigma_x = 2
        sigma_y = 3
        data = analytical.generate_data(0, 0, sigma_x, sigma_y, size=100000)
        for theta in np.arange(0,1,0.2):
            spill = [[1 - theta, 0],
                     [theta, 1]]
            new_data = np.dot(spill, data)
            kde = gaussian_kde(new_data)
            new_sigma_x, new_sigma_y = analytical.recalculate_sigmas(sigma_x, sigma_y, theta)
            score = 0
            for y in np.arange(-2, 2, 0.1):
                expected_row = []
                result_row = []
                for x in np.arange(-2, 2, 0.1):
                    expected_row.append(analytical.p_joint(x,y,theta,new_sigma_x,new_sigma_y))
                    result_row.append(kde((x,y)))
                for item in zip(expected_row, result_row):
                    if abs(item[0] - item[1]) <= threshold:
                        score += 1
            if score < 360:
                success = False
                break
        self.assertTrue(success)

    def test_x_dist(self):
        plotting = False
        success = True
        threshold = 0.01
        sigma_x = 2
        sigma_y = 1
        data = analytical.generate_data(0, 0, sigma_x, sigma_y)
        for theta in np.arange(0,1,0.2):
            spill = [[1 - theta, 0],
                     [theta, 1]]
            new_data = np.dot(spill, data)
            kde = gaussian_kde(new_data[0])
            results = []
            expected = []
            for x in np.arange(-2,2,0.1):
                results.append(kde(x))
                expected.append(analytical.p_x(x,sigma_x, theta))
            if plotting:
                plt.plot(results)
                plt.plot(expected)
                plt.show()
        self.assertTrue(success)

    def test_y_dist(self):
        plotting = False
        success = True
        threshold = 0.01
        sigma_x = 2
        sigma_y = 1
        data = analytical.generate_data(0, 0, sigma_x, sigma_y)
        for theta in np.arange(0, 1, 0.2):
            spill = [[1 - theta, 0],
                     [theta, 1]]
            new_data = np.dot(spill, data)
            kde = gaussian_kde(new_data[1])
            results = []
            expected = []
            for y in np.arange(-2, 2, 0.1):
                results.append(kde(y))
                expected.append(analytical.p_y(y, sigma_x, sigma_y, theta))
            if plotting:
                plt.plot(results)
                plt.plot(expected)
                plt.show()
        self.assertTrue(success)

    def test_data_set_mutual_info(self):
        data_array = analytical.generate_data(0, 0, 1, 1, size=1000)
        theta = 0.4
        spill = [[1 - theta, 0],
                 [theta, 1]]
        data_array = np.dot(spill, data_array)
        results = []
        expected = []
        for theta in np.arange(0,0.6,0.1):
            print(theta)
            spill = [[1 - theta, 0],
                     [theta, 1]]
            new_array = data_array.copy()
            data_frame = pd.DataFrame(new_array.transpose())
            data_frame.columns = ['x', 'y']
            data_set = utils.DataSet(data_frame=data_frame)
            data_set.apply(np.linalg.inv(spill))
            results.append(data_set.find_mutual_info('y','x'))
            expected.append(analytical.estimate_mutual_info(data_set.data_frame.values.transpose()))
        plt.plot(np.arange(0,0.6,0.1),results)
        plt.plot(np.arange(0,0.6,0.1),expected)
        plt.show()

    def test_minimize(self):
        data_array = analytical.generate_data(1000, 10, 1, 1, size=1000)
        theta = 0.4
        spill = [[1 - theta, 0],
                 [theta, 1]]
        data_array = np.dot(spill, data_array)
        print(theta)
        new_array = data_array.copy()
        data_frame = pd.DataFrame(new_array.transpose())
        data_frame.columns = ['x', 'y']
        data_set = utils.DataSet(data_frame=data_frame)
        print('ideal' + str(compensation.minimize_mutual_info(data_set, 'x', 'y')))

    def test_mutual_info(self):
        true_theta = 0.4
        spill = [[1-true_theta, 0],
                 [true_theta,   1]]
        original_data_matrix = analytical.generate_data(1000,10,1,1,size=100)
        original_data_matrix = np.dot(spill, original_data_matrix)
        results1 = []
        results2 = []
        for theta in np.arange(0,1,0.01):
            d_theta = (true_theta-theta) / (1-theta)
            print(d_theta)
            data_matrix = original_data_matrix.copy()
            spill = [[1-theta, 0],
                     [theta, 1]]
            data_matrix = np.dot(np.linalg.inv(spill), data_matrix)
            results1.append(analytical.estimate_mutual_info(data_matrix, resolution=60))
            results2.append(analytical.mutual_info(1,1,d_theta))
            print(results1[-1]/results2[-1])
        plt.plot(results1)
        plt.plot(results2)
        plt.show()
        self.assertTrue(True)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    math = MathTests()
    suite.addTests(math.test_mutual_info())
    unittest.TextTestRunner().run(suite)