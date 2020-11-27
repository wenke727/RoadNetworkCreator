import pandas as pd
import numpy as np
import math

class Line_vector(object):
    def __init__(self, item):
        self.x1 = item['x0']
        self.y1 = item['y0']
        self.x2 = item['x1']
        self.y2 = item['y1']
        self.v = self.vector()
        self.l = self.length()

    def vector(self):
        c = (self.x1 - self.x2, self.y1 - self.y2)
        return c

    def length(self):
        d = math.sqrt(pow((self.x1 - self.x2), 2) + pow((self.y1 - self.y2), 2))
        return d

def angle_bet_two_line(a, b):
    '''
    @input: pandas.core.series.Series, pandas.core.series.Series
    '''
    a = Line_vector(a)
    b = Line_vector(b)
    return np.arccos( np.dot(a.v, b.v) / (a.l*b.l) )/math.pi*180