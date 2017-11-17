'''
Created on Jun 13, 2017

@author: meike.zehlike
'''

import unittest
import measures2.statistical_tests as st
import pandas as pd
from data_structure.dataset import Dataset
import numpy as np


class Test(unittest.TestCase):

    def test_regression_slope_test(self):
        data = pd.DataFrame({'target': [1, 2, 3, 4, 5, 6, 7, 8],
                             'protected': [0, 1, 2, 3, 0, 1, 2, 3]})

        dataset = Dataset(data)

        st.regression_slope_test(dataset, 'target', 'protected')

    def test_two_proportion_z_test(self):
        data = pd.DataFrame({'target': [1, 2, 3, 4, 5, 6, 7, 8],
                             'protected': [0, 1, 2, 3, 0, 1, 2, 3]})

        dataset = Dataset(data)

        st.two_proportion_z_test(dataset, 'target', 'protected')

if __name__ == "__main__":
    unittest.main()

