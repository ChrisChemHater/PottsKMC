#pragma once
#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <utility>
#include <numeric>
#include <string>
#include <stdexcept>
#include <time.h>
#include <tclap/CmdLine.h>
#define _VERSION "0.1"
#define GAMMA 4
using namespace std;

typedef mt19937 random_engine;
struct Neighbors {
	Neighbors(int u, int d, int l, int r);
	int u;
	int d;
	int l;
	int r;
};

class FlipCount
{
public:
	FlipCount();
	FlipCount(int size);
	FlipCount(FlipCount&& _flipCount);
	FlipCount& operator=(FlipCount&& _flipCount);
	void resize(int size);
	~FlipCount();
	int size;
	bool* flipAllows;  // size = (2 * gamma + 1) * (2 * q - 1), if the flip class exists
	bool have(int flipInd);
};

struct Frame
{
	Frame();
	int i;  // row
	int j;  // col
	int nextSpin;  // sigma[i, j] change to nextSpin
	double dt;  // waiting time
	double E;  // Energy
};

class PottsModel
{
public:
	PottsModel(int N, int M, int q, double J, double h, double T, double tau, int randomSeed);
	~PottsModel();
	const int N, M;  // shape
	const int modelSize;
	const double J, h, T;  // Energy factors
	const int q;  // number of states
	const int flipClassNum;
	const double tau;  // time-scaling factor
	const int randomSeed;
	random_engine rng;
	bool initialized;

	double* DeltaHs;  // size = (2 * gamma + 1) * (2 * q - 1), deltaH for every flip classes
	int* sigma;  // N * M matrix
	Frame frame;  // 

	void initialize();  // initialize any state, and make the first suggestion
	void step();  // change state as suggested, and make new suggestion
	double getEnergy();  // the method calculate energy based on current sigma matrix
private:
	const int nbrCNum;
	const int spinCNum;
	FlipCount* _flipCountMap;  // N * M, flipCount on every atom
	int* _flipCountPerClass;  // (2 * gamma + 1) * (2 * q - 1), number of allowed flip of every flip class
	double* _Pflips;  // (2 * gamma + 1) * (2 * q - 1), probability of each flip class
	int* _candAtomInds;  // N * M, container of atoms candidates.
	double _deltaE; // (2 * gamma + 1) * (2 * q - 1), delta E of each flip class
	double _cdeltaE;  // The compensation item in Kahan accumulation.

	void initializeSigma();
	void initializeDeltaHs();
	void initializeFlipCounts();
	void initializePflips();
	void updateFCandPflips();  // update FlipCountMap and Pflips after update state
	void updateEnergy();

	int getEqualNbrNum(int n, int m);
	int getEqualNbrNum(int n, int m, int spin);
	void calAtomFlipCount(int n, int m);
	void calAtomFlipCount(int n, int m, int spin);
	Neighbors getNeighbors(int n, int m);

	void suggest(); // make new suggestion: flip which atom to which spin, and calculate waiting time
	double calWaitingTime();
};

int choiceProb(double* prob, int size, random_engine& rng);
int choice(int* choiceList, int size, random_engine& rng);

pair<int*, Frame*> simulate(int steps, int N, int M, int q, double J, double h, double T, double tau, int randomSeed, int recordFreq, bool silent);
pair<int*, Frame*> simulate(int steps, int N, int M, int q, double J, double h, double T, double tau, int randomSeed, Frame* traj, int recordFreq, bool silent);
void simulate(int steps, int N, int M, int q, double J, double h, double T, double tau, int randomSeed, ostream& _initStr, ostream& _traj, int recordFreq, bool silent);

enum TrajType
{
	OnlyChange,
	FullState
};

ostream& operator<<(ostream& _ostr, const Frame& frame);
ostream& writeTrajHead(ostream& _ostr, TrajType _type, int* sigma, int N, int M);
ostream& writeTraj(ostream& _ostr, const Frame* traj, int trajSize);
ostream& writeState(ostream& _ostr, int* sigma, int _size);
ostream& writeState(ostream& _ostr, int* sigma, int N, int M);
