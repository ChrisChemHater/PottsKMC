#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
question (c)
Study the correlation length xi as the function of T when h = 0.
"""
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from typing import List, Tuple

sys.path.append("./")
from api.experiments import PottsResult
from multiprocessing import Pool

def get_job_names(jobFile:str) -> List[str]:
	encoding = "gbk" if sys.platform == "win32" else "utf-8"
	with open(jobFile, "rt", encoding=encoding) as r:
		job_names = re.findall(r"(?<=--job-name )\S+", r.read())
	return job_names

def calculate_L(jobName:str, path_to_outfiles:str, qcDir:str) -> tuple:
	result = PottsResult(jobName, path_to_outfiles)
	res = (
		result.params['T'],
		result.get_lambda(),
		fit_xi(*result.get_lambda())
	)

	# 质控，绘制拟合线
	Lamb0, Lamb0_std, xi, xi_std = res[2]
	if np.isnan(Lamb0) or np.isnan(xi):
		fig, ax = plt.subplots(dpi=150)
		result.plot_lambda(ax)
	else:
		validCorr = _cutoff_x1(*result.get_lambda())
		ks = np.arange(len(validCorr))

		fig, ax = plt.subplots(dpi=150)
		ax.plot(ks, np.log(validCorr), 'k-', label="simulation")
		ax.plot(ks, np.log(Lamb0) - ks / xi, 'k--', label="fitted")
		ax.set_xlabel("Distance")
		ax.set_ylabel("Log(spatial correlation)")
		ax.legend()
	
	fig.savefig(os.path.join(qcDir, f"Corr-k_{jobName}.png"))
	
	return res

def plot_L(df:pd.DataFrame, spatialCorr:np.ndarray, spatialCorr_std:np.ndarray) -> tuple:
	temperatures = df['T'].values
	fig, ax = plt.subplots(dpi=150)
	ks = np.arange(spatialCorr.shape[1])
	cmap = plt.cm.get_cmap('jet')
	colors_ = cmap(np.linspace(0., 1., len(temperatures)))
	for color, temp, Lambda in zip(colors_, temperatures, spatialCorr):
		ax.plot(ks, Lambda, lw=0.7, color=color, label=f"T = {temp:.3f}")
	ax.set_xlabel("Distance")
	ax.set_ylabel("Spatial Correlation")
	plt.colorbar(plt.cm.ScalarMappable(
		norm=colors.Normalize(vmin=temperatures.min(), vmax=temperatures.max()),
		cmap=cmap), ax=ax)
	# ax.legend()
	return fig, ax

def _cutoff_x1(spatialCorr: np.ndarray, spatialCorr_std: np.ndarray) -> np.ndarray:
	# 截断，仅保留前面绝对值大于0的项和变异系数小于1/3的项
	validCorr = []
	CVs = spatialCorr_std / spatialCorr
	for corr, cv in zip(spatialCorr, CVs):
		if corr <= 0 or cv > 1/3:
			break
		validCorr.append(corr)
	return np.array(validCorr)

def fit_xi(spatialCorr: np.ndarray, spatialCorr_std: np.ndarray) -> tuple:
	# 截断
	validCorr = _cutoff_x1(spatialCorr, spatialCorr_std)
	ks = np.arange(len(validCorr))
	# 如果有效的数据量为0，报错
	if len(ks) == 0:
		return np.nan, np.nan, np.nan, np.nan
	if len(ks) == 1:
		return validCorr[0], spatialCorr_std[0], np.nan, np.nan

	# 固定截距为Corr[0], 回归，得到xi(负斜率倒数)
	try:
		# model = LinearRegression(fit_intercept=True).fit(ks.reshape(-1, 1), np.log(validCorr))
		# Lamb0 = np.exp(model.intercept_)
		# xi = - 1 / model.coef_[0]
		res = OLS(np.log(validCorr) - np.log(validCorr[0]), ks.reshape(-1, 1), hasconst=False).fit(use_t=True)
		Lamb0 = validCorr[0]
		Lamb0_std = spatialCorr_std[0]
		xi = -1 / res.params[0]
		xi_std = res.cov_params()[0, 0] ** 0.5 / xi ** 2
	except Exception as e:
		print(e)
		Lamb0, Lamb0_std, xi, xi_std = np.nan, np.nan, np.nan, np.nan
	return Lamb0, Lamb0_std, xi, xi_std

def plot_xi(df:pd.DataFrame) -> tuple:
	# 绘制Lamb0-T, xi-T图像
	temperatures = df['T'].values
	Lamb0s = df['Lamb0'].values
	Lamb0s_std = df['Lamb0_std'].values
	xis = df['Xi'].values
	xis_std = df['Xi_std'].values

	fig, (axLamb0, axXi) = plt.subplots(nrows=2, sharex=True, dpi=150)
	LambIdx = np.logical_not(np.logical_or(np.isnan(Lamb0s), np.isnan(Lamb0s_std)))
	axLamb0.errorbar(temperatures[LambIdx], Lamb0s[LambIdx], Lamb0s_std[LambIdx], fmt='k-', capsize=3, label=r"$\Lambda_0$")
	axLamb0.legend()
	axLamb0.set_ylabel(r"$\Lambda_0$")

	xiIdx = np.logical_not(np.logical_or(np.isnan(xis), np.isnan(xis_std)))
	axXi.errorbar(temperatures[xiIdx], xis[xiIdx], xis_std[xiIdx], fmt='k-', capsize=3, label=r"$\xi$")
	axXi.set_ylabel(r"Correlation Length")
	axXi.set_xlabel(r"Temperature")
	axXi.legend()
	return fig, (axLamb0, axXi)

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
		task = pool.apply_async(calculate_L, (jobName, path_to_outfiles, qcDir))
		tasks.append(task)
	results = [task.get() for task in tasks]
	
	# 存储spatialCorr结果
	temperatures = np.array([result[0] for result in results])
	Lambdas = np.r_[[result[1][0] for result in results]]
	Lambdas_std = np.r_[[result[1][1] for result in results]]
	Lambdas_fitMatrix = np.array([result[2] for result in results])
	df = pd.DataFrame(np.c_[temperatures.reshape(-1, 1), Lambdas_fitMatrix], columns=['T', 'Lamb0', 'Lamb0_std', 'Xi', 'Xi_std'])
	df.to_csv(os.path.join(resFileDir, "result_fitted_xi.csv"), index=False, encoding="utf-8")
	np.savez(os.path.join(resFileDir, "result_spatial_corr.npz"), spatialCorr=Lambdas, spatialCorr_std=Lambdas_std)

	# 绘制SpatialCorr-Distance图像
	fig, ax = plot_L(df, Lambdas, Lambdas_std)
	fig.savefig(os.path.join(graphDir, "spatialCorr-T.png"))
	plt.close(fig)

	# 绘制Lamb0-T, xi-T图像
	fig, axes = plot_xi(df)
	fig.savefig(os.path.join(graphDir, "Xi-T.png"))
	plt.close(fig)

def main():
	jobFile = "batch.ps1" if sys.platform == "win32" else "batch.sh"
	outDirs = ["outFiles/temperature_q3", "outFiles/temperature_q10"]
	for outDir in outDirs:
		analyse(get_job_names(os.path.join(outDir, jobFile)), outDir)

def test():
	# df = pd.read_csv(r"outFiles\temperature_q3\results\result_spatial_corr.csv", encoding="utf-8")
	# fig, ax = plot_L(df)
	# plt.show()
	calculate_L("Potts_q3_T0.315", r"outFiles\temperature_q3", r"outFiles\temperature_q3\results\qc")

def test1():
	# spatialFiles = np.load(r"outFiles\temperature_q3\results\result_spatial_corr.npz")
	# spatialCorr = spatialFiles['spatialCorr']
	# spatialCorr_std = spatialFiles['spatialCorr_std']
	df = pd.read_csv(r"outFiles\temperature_q3\results\result_fitted_xi.csv")

	plot_xi(df)
	plt.show()

if __name__ == "__main__":
	main()