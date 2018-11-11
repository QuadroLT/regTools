import os
import unittest
import pandas as pd
# from regTools import Regression
from regTools.regression import *


class RegModelTestCase(unittest.TestCase):

    def setUp(self):
        os.chdir('..')
        data = pd.read_csv('./datasets/11843-2ex2.csv', sep=';')
        self.inst = Regression(data, 'CONC', 'SIG')
        os.chdir('tests')

    def test__levene_test(self):
        self.assertLess(self.inst._levene_test(), 0.05)

    def test__get_weights(self):
        c, d = self.inst._get_weights()
        self.assertEqual('{:.2f}'.format(c), '4.46')
        self.assertEqual('{:.4f}'.format(d), '0.1501')

    def test__fit(self):
        a, b = self.inst.model.params
        self.assertEqual('{:.3f}'.format(a), '12.218')
        self.assertEqual('{:.5f}'.format(b), '1.52727')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
