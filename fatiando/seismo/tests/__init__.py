"""
Test suite for the fatiando.gravity package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 10-Sep-2010'

import unittest

import fatiando.seismo.tests.traveltime

def suite(label='fast'):

    testsuite = unittest.TestSuite()
    
    testsuite.addTest(fatiando.seismo.tests.traveltime.suite(label))
    
    return testsuite


if __name__ == '__main__':
    
    unittest.main(defaultTest='suite')