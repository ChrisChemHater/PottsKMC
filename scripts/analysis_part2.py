#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
question (b)
Plot and analyse magnetization as a function of h
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

def calculate_magnetization(jobName:str, path_to_outfiles:str, qcDir:str) -> tuple:
	result = PottsResult(jobName, path_to_outfiles)
	res = (
		result.params['T'],
		result.params['B'],
		*result.get_average_energy(),
		*result.get_magnetication()
	)

	# plot QC (Quality Control)
	fig, ax = plt.subplots(dpi=150)
	result.plot_energy(ax)
	fig.savefig(os.path.join(qcDir, f"Energy-time_{result.params['jobName']}.png"))
	plt.close(fig)

	return res

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
		task = pool.apply_async(calculate_magnetization, (jobName, path_to_outfiles, qcDir))
		tasks.append(task)
	results = np.array([task.get() for task in tasks])
	
	df = pd.DataFrame(results, columns=["T", "h", "Energy", "Energy_std", "Magnetization", "Magnetization_std"])
	df.sort_values(by=["T", "h"], inplace=True)
	df.to_csv(os.path.join(resFileDir, "result.csv"), index=False, encoding="utf-8")

	plot_result_file(os.path.join(resFileDir, "result.csv"), graphDir)

def plot_result_file(resultFile:str, graphDir):
	df = pd.read_csv(resultFile, encoding="utf-8")
	# 将不同温度的轨迹分开
	matrix = df["Magnetization"].values.reshape(3, -1)
	matrix_std = df["Magnetization_std"].values.reshape(3, -1)
	temperatures = df["T"].values[::matrix.shape[1]]
	Bs = df["h"].values[:matrix.shape[1]]

	fig, ax = plt.subplots(dpi=150)
	for tidx, temp in enumerate(temperatures):
		# ax.scatter(Bs, matrix[tidx, :], marker='o', label=f"T = {temp:.3f}")
		ax.errorbar(Bs, matrix[tidx, :], matrix_std[tidx, :], capsize=3, label=f"T = {temp:.3f}")
	ax.vlines(0., *ax.get_ylim(), colors="k", linestyles="dashed")
	ax.set_xlabel("external magnetic field intensity")
	ax.set_ylabel("Magnetization")
	ax.legend()
	fig.savefig(os.path.join(graphDir, "M-h.png"))
	plt.close(fig)

def test():
	calculate_magnetization("Potts_q3_B_-1.000_T0.497", "outFiles/magnetic_q3")

def main():
	jobFile = "batch.ps1" if sys.platform == "win32" else "batch.sh"
	outDirs = ["outFiles/magnetic_q3", "outFiles/magnetic_q10"]
	for outDir in outDirs:
		analyse(get_job_names(os.path.join(outDir, jobFile)), outDir)

if __name__ == "__main__":
	# plot_result_file(r"outFiles\magnetic_q3\results\result.csv", r"outFiles\magnetic_q3\results\graph")
	# plot_result_file(r"outFiles\magnetic_q10\results\result.csv", r"outFiles\magnetic_q10\results\graph")
	main()