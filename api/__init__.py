#-*- coding:utf-8 -*-

def get_theoretical_Tc(J:float, q:int, gamma:int=4, kB:float=1.):
	return J * gamma / (kB * q)