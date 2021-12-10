#include "potts.h"

Frame::Frame(){}

FlipCount::FlipCount() : size(0), flipAllows(nullptr) {}

FlipCount::FlipCount(int size):size(size)
{
	flipAllows = new bool[size];
}

FlipCount::FlipCount(FlipCount&& _flipCount):size(_flipCount.size), flipAllows(_flipCount.flipAllows)
{
	_flipCount.flipAllows = nullptr;
}

FlipCount& FlipCount::operator=(FlipCount&& _flipCount)
{
	size = _flipCount.size;
	flipAllows = _flipCount.flipAllows;
	_flipCount.flipAllows = nullptr;
	return *this;
}

void FlipCount::resize(int newSize)
{
	this->size = newSize;
	delete[] flipAllows;
	flipAllows = new bool[size];
}

FlipCount::~FlipCount()
{
	delete[] flipAllows;
}

bool FlipCount::have(int flipInd)
{
	if (flipInd >= size)
		throw out_of_range("Out of range in FlipCount");
	return flipAllows[flipInd];
}

PottsModel::PottsModel(int N, int M, int q, double J, double h, double T, double tau, int randomSeed)
	: N(N), M(M), q(q), J(J), h(h), T(T), tau(tau), randomSeed(randomSeed), rng(randomSeed), initialized(false),
	modelSize(N * M), flipClassNum((2 * GAMMA + 1) * (2 * q - 1)), _deltaE(0.), _cdeltaE(0.), _ct(0.),
	DeltaHs(nullptr), sigma(nullptr), _flipCountMap(nullptr), _flipCountPerClass(nullptr), _Pflips(nullptr), _candAtomInds(nullptr),
	nbrCNum(2 * GAMMA + 1), spinCNum(2 * q - 1) {}

PottsModel::~PottsModel()
{
	if (initialized) {
		delete[] DeltaHs;
		delete[] sigma;
		delete[] _flipCountMap;
		delete[] _flipCountPerClass;
		delete[] _Pflips;
		delete[] _candAtomInds;
	}
}

void PottsModel::initialize()
{
	initializeSigma();
	initializeDeltaHs();
	initializeFlipCounts();
	initializePflips();
	_candAtomInds = new int[modelSize];
	frame.E = getEnergy();
	frame.t = 0.;
	suggest();
	initialized = true;
}

void PottsModel::step()
{
	sigma[frame.i * M + frame.j] = frame.nextSpin;
	updateFCandPflips();
	updateEnergy();
	updateTime();
	suggest();
}

double PottsModel::getEnergy()
{
	int bondNum = 0;
	int totalSpin = accumulate(sigma, sigma + modelSize, 0);
	for (int n = 0; n < N; ++n) {
		for (int m = 0; m < M; ++m) {
			bondNum += getEqualNbrNum(n, m);
		}
	}
	return bondNum / 2 * (-J) - h * totalSpin;
}

void PottsModel::initializeSigma()
{
	sigma = new int[N * M];
	uniform_int_distribution<int> dist(1, q);
	for (int i = 0; i < N * M; ++i)
		sigma[i] = dist(rng);
}

void PottsModel::initializeDeltaHs()
{
	DeltaHs = new double[nbrCNum * spinCNum];
	for (int k = 0; k < nbrCNum; ++k) {
		for (int l = 0; l < spinCNum; ++l) {
			DeltaHs[k * spinCNum + l] = -J * (k - GAMMA) - h * (l + 1 - q);
		}
	}
}

void PottsModel::initializeFlipCounts()
{
	_flipCountMap = new FlipCount[N * M];
	for (int i = 0; i < N * M; ++i) {
		_flipCountMap[i].resize(nbrCNum * spinCNum);
	}
	for (int n = 0; n < N; ++n) {
		for (int m = 0; m < M; ++m) {
			calAtomFlipCount(n, m);
		}
	}
}

void PottsModel::initializePflips()
{
	_flipCountPerClass = new int[nbrCNum * spinCNum];
	_Pflips = new double[nbrCNum * spinCNum];
	for (int i = 0; i < nbrCNum * spinCNum; ++i) {
		_flipCountPerClass[i] = 0;
		for (int j = 0; j < N * M; ++j) {
			if (_flipCountMap[j].have(i))
				++_flipCountPerClass[i];
		}
		_Pflips[i] = min<double>(exp(-DeltaHs[i] / T), 1.) * _flipCountPerClass[i];
	}
}

void PottsModel::updateFCandPflips()
{
	Neighbors nbrs = getNeighbors(frame.i, frame.j);
	for (int atomInd : {frame.i * M + frame.j, nbrs.u, nbrs.d, nbrs.l, nbrs.r}) {
		for (int i = 0; i < nbrCNum * spinCNum; ++i) {
			if (_flipCountMap[atomInd].have(i)) {
				_flipCountMap[atomInd].flipAllows[i] = false;
				--_flipCountPerClass[i];
				_Pflips[i] = min(1., exp(-DeltaHs[i] / T)) * _flipCountPerClass[i];
			}
		}
		calAtomFlipCount(atomInd / M, atomInd % M);
		for (int i = 0; i < nbrCNum * spinCNum; ++i) {
			if (_flipCountMap[atomInd].have(i)) {
				++_flipCountPerClass[i];
				_Pflips[i] = min(1., exp(-DeltaHs[i] / T)) * _flipCountPerClass[i];
			}
		}
	}
}

void PottsModel::updateEnergy()
{
	// Kahan accumulation
	double y = _deltaE - _cdeltaE;
	double t = frame.E + y;
	_cdeltaE = t - frame.E - y;
	frame.E = t;
}

void PottsModel::updateTime()
{
	// Kahan accumulation
	double y = frame.dt - _ct;
	double t = frame.t + y;
	_ct = t - frame.t - y;
	frame.t = t;
}

int PottsModel::getEqualNbrNum(int n, int m)
{
	return getEqualNbrNum(n, m, sigma[n * M + m]);
}

int PottsModel::getEqualNbrNum(int n, int m, int spin)
{
	Neighbors nbrs = getNeighbors(n, m);
	int equalNbrNum = \
		(int)(sigma[nbrs.u] == spin)\
		+ (int)(sigma[nbrs.d] == spin)\
		+ (int)(sigma[nbrs.l] == spin)\
		+ (int)(sigma[nbrs.r] == spin);
	return equalNbrNum;
}

void PottsModel::calAtomFlipCount(int n, int m)
{
	calAtomFlipCount(n, m, sigma[n * M + m]);
}

void PottsModel::calAtomFlipCount(int n, int m, int spin)
{
	int atomInd = n * M + m;
	for (int j = 0; j < _flipCountMap[atomInd].size; ++j)
		_flipCountMap[atomInd].flipAllows[j] = false;
	int equalNbrNum = getEqualNbrNum(n, m, spin);
	for (int s = 1; s <= q; ++s) {
		if (s == spin) continue;
		_flipCountMap[atomInd].flipAllows[(getEqualNbrNum(n, m, s) + GAMMA - equalNbrNum) * spinCNum + s + q - 1 - sigma[atomInd]] = true;
	}
}

Neighbors PottsModel::getNeighbors(int n, int m)
{
	return Neighbors(
		((n + N - 1) % N) * M + m,
		((n + 1) % N) * M + m,
		n * M + (m - 1 + M) % M,
		n * M + (m + 1) % M
	);
}

void PottsModel::suggest()
{
	// decide flip class
	int flipInd = choiceProb(_Pflips, flipClassNum, rng);

	// decide flip atom
	int candAtomNum = 0;
	for (int i = 0; i < modelSize; ++i) {
		if (_flipCountMap[i].have(flipInd))
			_candAtomInds[candAtomNum++] = i;
	}
	int atomInd = choice(_candAtomInds, candAtomNum, rng);
	frame.i = atomInd / M;
	frame.j = atomInd % M;
	frame.nextSpin = sigma[atomInd] + flipInd % spinCNum - q + 1;
	calWaitingTime();
	_deltaE = DeltaHs[flipInd];
}

double PottsModel::calWaitingTime()
{
	double totalProb = accumulate(_Pflips, _Pflips + flipClassNum, 0.);
	frame.dt = exponential_distribution<double>(totalProb / tau)(rng);
	return frame.dt;
}

int choiceProb(double* prob, int size, random_engine& rng)
{
	double totalProb = accumulate(prob, prob + size, 0.);
	double R = uniform_real_distribution<double>(0., totalProb)(rng);
	for (int i = 0; i < size; ++i) {
		if ((R = R - prob[i]) <= 0) return i;
	}
	return size - 1;
}

int choice(int* choiceList, int size, random_engine& rng)
{
	int R = uniform_int_distribution<int>(0, size - 1)(rng);
	return choiceList[R];
}

Neighbors::Neighbors(int u, int d, int l, int r) :u(u), d(d), l(l), r(r) {}

ostream& operator<<(ostream& _ostr, const Frame& frame)
{
	_ostr << frame.i << ' ' << frame.j << ' ' << frame.nextSpin << ' ' << frame.t << ' ' << frame.dt << ' ' << frame.E << '\n';
	return _ostr;
}

ostream& writeTrajHead(ostream& _ostr, TrajType _type, int* sigma, int N, int M)
{
	switch (_type)
	{
	case OnlyChange:
		_ostr << "# type = OnlyChange\n";
		_ostr << "# Initial State = ";
		writeState(_ostr, sigma, N * M);
		_ostr << '\n';
		_ostr << "# row col nextSpin t dt Energy\n";
		break;
	case FullState:
		_ostr << "# type = FullState\n";
		_ostr << "# REMARK: each frame is one line, contains stepIndex, state, t, dt and Energy\n";
		break;
	default:
		break;
	}
	return _ostr;
}

ostream& writeState(ostream& _ostr, int* sigma, int _size)
{
	for (int i = 0; i < _size; ++i) {
		_ostr << sigma[i] << ' ';
	}
	return _ostr;
}

ostream& writeState(ostream& _ostr, int* sigma, int N, int M)
{
	for (int n = 0; n < N; ++n) {
		for (int m = 0; m < M; ++m) {
			_ostr << sigma[n * M + m] << ' ';
		}
		_ostr << '\n';
	}
	return _ostr;
}