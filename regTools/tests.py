import unittest
import pandas as pd
from regTools  import Regression


class RegModelTestCase(unittest.TestCase):


    def setUp(self):
       data = pd.read_csv('11843.csv', sep=';')
       self.inst = Regression(data, 'CONC', 'SIG', model='WLS')

    def test_is_homoscedastic(self):
        self.assertEqual(self.inst.is_homoscedastic(), False)

    def test_get_weights(self):
        c, d = self.inst.get_weights()
        self.assertEqual('{:.2f}'.format(c), '4.46')
        self.assertEqual('{:.4f}'.format(d), '0.1501')
    def test_fit(self):
        a, b = self.inst.reg_model.params
        self.assertEqual('{:.3f}'.format(a), '12.218')
        self.assertEqual('{:.5f}'.format(b), '1.52727')

    def test_plot(self):
        a = self.inst
        plot_model(a, xlabel='conc, mg/ml', ylabel=u'$\mu$g/kg')


    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
