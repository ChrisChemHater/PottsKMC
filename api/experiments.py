#-*- coding:utf-8 -*-
import re
import os
from typing import List, Union, Tuple
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

DEFAUT_PARAMS = {
	"N": 1,
	"M": 1,
	"q": 2,
	"steps": 2,
	"recordFreq": 1,
	"randomSeed": 1234,
	"J": 1.,
	"B": 0.,
	"T": 1.,
	"tau": 10.,
	"jobName": "potts"
}

PARAM_TRANS = {
	"N": int,
	"M": int,
	"q": int,
	"steps": int,
	"recordFreq": int,
	"randomSeed": int,
	"J": float,
	"B": float,
	"T": float,
	"tau": float,
	"jobName": str
}

PARAM_FORMATS = {
	"N": "-N {}",
	"M": "-M {}",
	"q": "-q {}",
	"steps": "--steps {}",
	"recordFreq": "--record-freq {}",
	"randomSeed": "--random-seed {}",
	"J": "-J {}",
	"B": "-B {}",
	"T": "-T {}",
	"tau": "--tau {}",
	"jobName": "--job-name {}"
}

POTTSKMC_PATH = os.path.abspath(os.environ.get("PottsKMC", "./PottsKMC"))

class PottsExperiment:
	def __init__(self) -> None:
		self.params = {}
	@classmethod
	def from_log(cls, logFile: str):
		expr = cls()
		with open(logFile, "rt") as r:
			while True:
				line = r.readline()
				if line == '':
					raise ValueError("Unvalid log file")
				if line == "PARAMS\n":
					break
			rawParams = re.findall(r"^\w+\s*=\s*\w+.*$",
								   r.read(),
								   flags=re.MULTILINE)
		for rawParam in rawParams:
			k, v = [s.strip() for s in rawParam.split('=')]
			expr.params[k] = PARAM_TRANS[k](v)
		# check if there is something missing
		if set(expr.params.keys()) != set(PARAM_TRANS.keys()):
			raise ValueError(f"Missing parameter in log file: {logFile}")
		return expr

	@classmethod
	def from_dict(cls, params: dict):
		expr = cls()
		params_ = DEFAUT_PARAMS.copy()
		params_.update(params)
		for k, v in params_.items():
			if k in DEFAUT_PARAMS:
				expr.params[k] = PARAM_TRANS[k](v)
		return expr
	
	def dict_to_run_args(self, params:dict):
		args = [s.format(params[k]) for k, s in PARAM_FORMATS.items()]
		args = [POTTSKMC_PATH] + args + ["--quiet"]
		return args
	
	def get_command(self, params: dict=None) -> str:
		if params is None:
			params_ = self.params
		else:
			params_ = self.params.copy()
			params_.update(params)
		args = [s.format(params_[k]) for k, s in PARAM_FORMATS.items()]
		args = [POTTSKMC_PATH] + args + ["--quiet"]
		return ' '.join(args)

class PottsResult:
	def __init__(self, jobName:str, path:str='./') -> None:
		self.experiment = PottsExperiment.from_log(os.path.join(path, f"{jobName}.log"))
		self.params = self.experiment.params
		self.kB = 1.
		self.path = path
		self.properties = {}

		self.load_traj(os.path.join(path, f"{jobName}.traj"))
	
	def load_traj(self, trajFile):
		with open(trajFile, "rt") as r:
			trajType = re.search(r"FullState|OnlyChange", r.readline())
		if trajType is None:
			raise KeyError(f"Invalid type mark in the 1st line of {trajFile}")
		trajType = trajType.group()
		if trajType == "FullState":
			self._load_fullstate_traj(trajFile)
			self.totalSampleTime = self.trajDt.sum()
		else:
			self._load_onlychange_traj(trajFile)
			self.totalSampleTime = self.totalTime
	
	@property
	def totalTime(self) -> float:
		return self.trajT[-1] + self.trajDt[-1] - self.trajT[0]
	
	def _load_onlychange_traj(self, trajFile:str):
		N, M = self.params['N'], self.params['M']
		with open(trajFile, "rt") as r:
			r.readline()
			k, v = r.readline().strip().split('=')
			if not "Initial State" in k:
				raise KeyError(f"Invalid Initial State entry in the 2nd line of {trajFile}")
		initialState = np.array(map(int, v.strip().split()), dtype=np.int64).reshape(N, M)
		rawSuggests = np.loadtxt(trajFile)
		self.trajIdx = np.arange(len(rawSuggests))
		self.traj = np.repeat(initialState[np.newaxis, ...], len(rawSuggests), axis=0)
		suggests = rawSuggests[:-1,:3].astype(np.int64)  # suggest flip decide the state of next frame, so length is N - 1
		self.trajT = rawSuggests[:,3].astype(np.double)
		self.trajDt = rawSuggests[:,4].astype(np.double)
		self.trajEnergy = rawSuggests[:,5].astype(np.double)
		for step, (i, j, nextSpin) in enumerate(suggests):
			self.traj[step + 1] = self.traj[step]
			self.traj[step + 1, i, j] = nextSpin
	
	def _load_fullstate_traj(self, trajFile:str):
		rawTraj = np.loadtxt(trajFile)
		N, M = self.params['N'], self.params['M']
		self.trajIdx = rawTraj[:, 0].astype(np.int64)
		self.traj = rawTraj[:, 1:N*M + 1].astype(np.int64).reshape(-1, N, M)
		self.trajT = rawTraj[:, N*M+1].astype(np.double)
		self.trajDt = rawTraj[:, N*M+2].astype(np.double)
		self.trajEnergy = rawTraj[:, N*M+3].astype(np.double)
	
	def recalculate_average_energy(self) -> float:
		self.properties["aveEnergy"] = (self.trajEnergy * self.trajDt).sum() / self.totalSampleTime
		return self.properties["aveEnergy"]

	def get_average_energy(self) -> float:
		if "aveEnergy" in self.properties:
			return self.properties["aveEnergy"]
		return self.recalculate_spatial_correlation()
	
	def recalculate_heat_capacity(self) -> float:
		N, M = self.params['N'], self.params['M']
		varH = ((self.trajEnergy - self.recalculate_average_energy()) ** 2 * self.trajDt).sum() / self.totalSampleTime
		self.properties["heatCapacity"] = self.kB / (N * M * self.params['T'] ** 2) * varH
		return self.properties["heatCapacity"]
	
	def get_heat_capacity(self) -> float:
		if "heatCapacity" in self.properties:
			return self.properties["heatCapacity"]
		return self.recalculate_spatial_correlation()
	
	def recalculate_magnetization(self) -> float:
		self.properties["magnetization"] = (self.traj.mean(axis=(1,2)) * self.trajDt).sum() / self.totalSampleTime
		return self.properties["magnetization"]
	
	def get_magnetication(self):
		if "magnetization" in self.properties:
			return self.properties["magnetization"]
		return self.recalculate_spatial_correlation()
	
	def recalculate_spatial_correlation(self, cutoff:Union[int, Tuple[int]] = None) -> np.ndarray:
		"""
		计算空间关联。
		Arguments:
			cutoff: 计算空间关联的最大尺寸，若为None，则计算到(N/2, M/2)
		Return:
			Corr, np.ndarray, shape = (Nmax, Mmax), in which Corr[i, j] is the spatial correlation between (0, 0) and (i, j)
		"""
		if cutoff is None:
			Nmax, Mmax = self.params['N'] // 2, self.params['M'] // 2
		elif np.isscalar(cutoff):
			Nmax, Mmax = cutoff, cutoff
		else:
			Nmax, Mmax = cutoff
		correlations = np.zeros((Nmax, Mmax), dtype=np.float64)
		for i in range(Nmax):
			for j in range(Mmax):
				correlations[i, j] = self._calculate_spatial_correlation(i, j)
		self.properties["spatialCorrelation"] = correlations
		return correlations
	
	def get_spatial_correlation(self, cutoff:Union[int, Tuple[int]] = None) -> np.ndarray:
		"""
		计算空间关联。
		Arguments:
			cutoff: 计算空间关联的最大尺寸，若为None，则计算到(N/2, M/2)
		Return:
			Corr: np.ndarray, shape = (Nmax, Mmax), in which Corr[i, j] is the spatial correlation between (0, 0) and (i, j)
		"""
		if cutoff is None:
			Nmax, Mmax = self.params['N'] // 2, self.params['M'] // 2
		elif np.isscalar(cutoff):
			Nmax, Mmax = cutoff, cutoff
		else:
			Nmax, Mmax = cutoff
		if "spatialCorrelation" in self.properties and (Nmax, Mmax) == self.properties["spatialCorrelation"].shape:
			return self.properties["spatialCorrelation"]
		return self.recalculate_spatial_correlation(cutoff)
	
	def _calculate_spatial_correlation(self, i:int, j:int) -> float:
		shiftedTraj = np.roll(np.roll(self.traj, -i, axis=1), -j, axis=2)
		aveSigma = (self.traj * self.trajDt[:, np.newaxis, np.newaxis]).sum(axis=0) / self.totalSampleTime
		aveSigmaShifted = np.roll(np.roll(aveSigma, -i, axis=0), -j, axis=1)
		corr = ((shiftedTraj * self.traj) * self.trajDt[:, np.newaxis, np.newaxis]).sum(axis=0) / self.totalSampleTime
		return (corr - aveSigma * aveSigmaShifted).mean()
	
	def recalculate_lambda(self, cutoff:int = None) -> np.ndarray:
		"""
		计算四重对称空间关联。即Lambda[k] = mean(Corr[0, k], Corr[0, -k], Corr[k, 0], Corr[-k, 0])
		由于算法中Corr[0, k] = Corr[0, -k], Corr[k, 0] = Corr[-k, 0]，实际计算中仅取Corr[0, k]和Corr[k, 0]的平均
		Arguments:
			cutoff: 计算对称关联的最大尺寸，若为None，则计算到min(N/2, M/2)
		Return:
			Lambda: np.ndarray, shape = (Nmax,)
		"""
		if cutoff is None:
			Nmax = min(self.params['N'] // 2, self.params['M'] // 2)
		VCorr = np.array([self._calculate_spatial_correlation(i, 0) for i in range(Nmax)])
		HCorr = np.array([self._calculate_spatial_correlation(0, j) for j in range(Nmax)])
		self.properties["Lambda"] = (VCorr + HCorr) / 2
		return self.properties["Lambda"]
	
	def get_lambda(self, cutoff:int = None) -> np.ndarray:
		"""
		计算四重对称空间关联。即Lambda[k] = mean(Corr[0, k], Corr[0, -k], Corr[k, 0], Corr[-k, 0])
		由于算法中Corr[0, k] = Corr[0, -k], Corr[k, 0] = Corr[-k, 0]，实际计算中仅取Corr[0, k]和Corr[k, 0]的平均
		Arguments:
			cutoff: 计算对称关联的最大尺寸，若为None，则计算到min(N/2, M/2)
		Return:
			Lambda: np.ndarray, shape = (Nmax,)
		"""
		if cutoff is None:
			Nmax = min(self.params['N'] // 2, self.params['M'] // 2)
		if ("Lambda" in self.properties) and Nmax == len(self.properties["lambda"]):
			return self.properties["Lambda"]
		return self.recalculate_lambda(cutoff)

	def plot_energy(self, ax):
		plot = ax.plot(self.trajT, self.trajEnergy, "-k", lw=0.5)
		ax.set_xlabel("Time")
		ax.set_ylabel("Energy")
		return plot
	
	def plot_lambda(self, ax):
		Lamb = self.properties["Lambda"]
		plot = ax.plot(np.arange(len(Lamb)), Lamb, "-k", lw=0.7)
		ax.plot(np.arange(len(Lamb)), Lamb, ".k", markersize=5.0)
		ax.set_xlabel("k (distance)")
		ax.set_ylabel("Spatial Correlation")
		return plot
	
	def run_all_analysis(self):
		self.recalculate_average_energy()
		self.recalculate_heat_capacity()
		self.recalculate_magnetization()
		self.recalculate_spatial_correlation()
		self.recalculate_lambda()

	def summary(self):
		print(self.properties)

	def __getitem__(self, slice_:slice):
		newItem = deepcopy(self)
		newItem.trajIdx = newItem.trajIdx[slice_]
		newItem.traj = newItem.traj[slice_]
		newItem.trajT = newItem.trajT[slice_]
		newItem.trajDt = newItem.trajDt[slice_]
		newItem.trajEnergy = newItem.trajEnergy[slice_]
		newItem.properties = {}
		newItem.totalSampleTime = newItem.totalTime if newItem.params["recordFreq"] == 1 else newItem.trajDt.sum()
		return newItem

def writeBatchFile(expr:PottsExperiment, paramList:List[dict], jobFile:str, batchSize:int=-1):
	"""
	如果batchSize有设定，则每batchSize个任务设置一个join点
	"""
	writer = open(jobFile, "wt", encoding="utf-8")
	writer.write("#!/usr/bin/env bash\n")
	for i, params in enumerate(paramList, start=1):
		writer.write(expr.get_command(params) + " &\n")
		if batchSize > 0 and i % batchSize == 0:
			writer.write("wait\n")
	writer.write("wait\n")
	writer.close()

def test():
	res = PottsResult("potts_N20_T5_q3")
	res.run_all_analysis()
	res.summary()

	fig, axes = plt.subplots(nrows=2, dpi=150)
	res.plot_energy(axes[0])
	res.plot_lambda(axes[1])
	plt.show()

if __name__ == "__main__":
	test()