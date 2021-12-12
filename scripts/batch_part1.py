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
	# 生成临界点附近的温度序列
	def genBatch(q:int):
		Tc = get_theoretical_Tc(expr.params['J'], q)
		temperatures = np.logspace(np.log10(Tc) - 0.5, np.log10(Tc) + 0.8, 40)
		additionParamList = [{'q': q, 'T': temp, "jobName": f"Potts_q{q}_T{temp:.3f}", "randomSeed": np.random.randint(0xffff)} for temp in temperatures]
		jobDir = f"outFiles/temperature_q{q}"
		if not os.path.isdir(jobDir):
			os.makedirs(jobDir)
		if sys.platform == "win32":
			writeBatchFilePws(expr, additionParamList, os.path.join(jobDir, "batch.ps1"), 10)
		else:
			writeBatchFile(expr, additionParamList, os.path.join(jobDir, "batch.sh"), 10)

	genBatch(3)
	genBatch(10)
