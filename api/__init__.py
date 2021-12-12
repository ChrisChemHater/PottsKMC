#-*- coding:utf-8 -*-
import numpy as np

def get_theoretical_Tc(J:float, q:int, kB:float=1.):
	return J / (kB * np.log(1 + np.sqrt(q)))