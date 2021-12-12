#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
生成batch.sh文件
"""
import os, sys
sys.path.append(os.path.abspath("./"))
from api.experiments import PottsExperiment, writeBatchFile, writeBatchFilePws
from api import get_theoretical_Tc
import numpy as np

if __name__ == "__main__":
	# 基础实验对象
	expr = PottsExperiment.from_dict({
		"N": 20,
		"M": 20,
		"q": 3,
		"steps": 5000000,
		"recordFreq": 500,
		"randomSeed": 1234,
		"J": 1.,
		"B": 0.,
		"T": 1.,
		"tau": 20.,
		"jobName": "potts"
	})
	def simu(q:int):
		Tc = get_theoretical_Tc(expr.params['J'], q)
		# 在Tc上中下取三个温度进行模拟
		temperatures = np.array([Tc / 2., Tc, Tc * 2])
		Bs = np.linspace(-1., 1., 21)
		additionParamList = [{'q': q, 'T': temp, 'B':B, "jobName": f"Potts_q{q}_B_{B:.3f}_T{temp:.3f}", "randomSeed": np.random.randint(0xffff)} for temp in temperatures for B in Bs]
		jobDir = f"outFiles/magnetic_q{q}"
		if not os.path.isdir(jobDir):
			os.makedirs(jobDir)
		# writeBatchFile(expr, additionParamList, os.path.join(jobDir, "batch.sh"), 10)
		writeBatchFilePws(expr, additionParamList, os.path.join(jobDir, "batch.ps1"), 10)

	simu(3)
	simu(10)

