#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
question (a)
Plot and analyse energy and specific heat as a function of temperature
"""
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
sys.path.append("./")
from api.experiments import PottsResult
from multiprocessing import Pool

def get_job_names(jobFile:str) -> List[str]:
	encoding = "gbk" if sys.platform == "win32" else "utf-8"
	with open(jobFile, "rt", encoding=encoding) as r:
		job_names = re.findall(r"(?<=--job-name )\S+", r.read())
	return job_names

def calculate_E_C_and_plotQC(jobName:str, path_to_outfiles:str, qcDir:str) -> tuple:
	result = PottsResult(jobName, path_to_outfiles)
	aveE = result.get_average_energy()
	hC = result.get_heat_capacity()
	res = (
		result.params['T'],
		aveE[0],
		aveE[1],
		hC[0],
		hC[1]
	)

	# plot QC (Quality Control)
	fig, ax = plt.subplots(dpi=150)
	result.plot_energy(ax)
	fig.savefig(os.path.join(qcDir, f"Energy-time_{result.params['jobName']}.png"))
	plt.close(fig)

	return res

def plot_E_T_C_T(df:pd.DataFrame):
	fig, axes = plt.subplots(nrows=2, sharex=True, dpi=150)
	axes[0].errorbar(df['T'], df['Energy'], df['Energy_std'], fmt="-k", capsize=2, label="Average Energy")
	axes[0].set_ylabel("Energy")
	axes[0].legend()
	axes[1].errorbar(df['T'], df['HeatCapacity'], df['HeatCapacity_std'], fmt="-k", capsize=2, label="HeatCapacity")
	axes[1].set_xlabel("Temperature")
	axes[1].set_ylabel("Heat Capacity")
	axes[1].legend()
	return fig, axes

def analyse(jobNames:List[str], path_to_outfiles:str) -> None:
	resFileDir = os.path.join(path_to_outfiles, "results")
	graphDir = os.path.join(resFileDir, "graph")
	if not os.path.isdir(graphDir):
		os.makedirs(graphDir)
	qcDir = os.path.join(resFileDir, "qc")
	if not os.path.isdir(qcDir):
		os.makedirs(qcDir)
	
	pool = Pool(10)
	tasks = []
	for jobName in jobNames:
		task = pool.apply_async(calculate_E_C_and_plotQC, (jobName, path_to_outfiles, qcDir))
		tasks.append(task)
	results = np.array([task.get() for task in tasks])
	
	df = pd.DataFrame(results, columns=["T", "Energy", "Energy_std", "HeatCapacity", "HeatCapacity_std"])
	df.to_csv(os.path.join(resFileDir, "result.csv"), index=False, encoding="utf-8")

	fig, ax = plot_E_T_C_T(df)
	fig.savefig(os.path.join(graphDir, "E-T_C-T.png"))
	plt.close(fig)

def test1():
	print(get_job_names("outFiles/temperature_q3/batch.sh"))

def main():
	jobFile = "batch.ps1" if sys.platform == "win32" else "batch.sh"
	outDirs = ["outFiles/temperature_q3", "outFiles/temperature_q10"]
	for outDir in outDirs:
		analyse(get_job_names(os.path.join(outDir, jobFile)), outDir)

if __name__ == "__main__":
	main()