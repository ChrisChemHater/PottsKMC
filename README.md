# PottsKMC

Simulate Pottsmodel with high efficiency using KMC method.

## Prerequisites

- A C++ compiler, MSVC and GCC have been tested
- Command line arguments parser [TCLAP 1.4](http://tclap.sourceforge.net/)
- Python >= 3.7
- NumPy, Pandas, statsmodels, matplotlib

## Compile

Enter `src/` directory, and run `make`. An excutable `PottsKMC` will be created in `src/`.

Run `PottsKMC --help` to get more information.

## Simulation

### Potts model setup

Periodic lattice with size `N * M`, each atom has `q` spin states (from 1 to q), the Hamitonian of the 
system is defined as:

$$
H=-J\sum_{<i,j>}\delta_{s_i s_j}-B\sum_i s_i
$$

In which $\sum_{<i,j>}$ means only take neighbor atoms into summation for each atom.

### Run simulation with PottsKMC

KMC simulation is to sample the canonical ensemble of a specific system. The trajectory is a series of states, each state has 
a waiting time sampled from an exponential distribution. The `PottsKMC` program take `-T <float>` as temperature, and `--tau <float>` 
as time scaling factor. The latter thretically has no effect on the ensemble average calculation, but may accutually influence the 
precision due to the limited precision of float number stored in machine.

As the autocorrelation is severe in KMC simulation, and the cost of writing trajectory file is considerably high, `PottsKMC` provided 
`--record-freq <int>` parameter to decide there should be how many steps between two record frames. And note that the standard error given 
by analysis tools is calculated under i.i.d assumption, which means that the std value is wrong when the autocorrelation is not negligible.

```
PottsKMC -N 20 -M 20 -q 3 --steps 5000000 --record-freq 500 --random-seed 38347 -J 1.0 -B 0.0 -T 0.8954755749646353 --tau 20.0 --job-name Potts_q3_T0.89548 --quiet
```

Three files will be produced in the work directory: Init file, Log file and Traj file, recording 
initial state, setup message and trajectory, respectively.

The program will produce a trajectory with `steps + 1` frames. Each frame is a record of system 
state, energy and starting time and waiting time.

## Trajectory Analysis

The simple analysis api is written in Python. `api.experiments.PottsResult` can automatically load the trajectory, 
and do the listed calculations:

- Average Energy
- Heat capacity
- Magnetization
- Spatial Correlation

Examples can be found in `scripts` directory.
