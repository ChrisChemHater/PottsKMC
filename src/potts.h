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
	bool* flipAllows;  // size = (2 * gamma + 1) * (2 * q - 1), ��¼һ��ԭ���Ƿ�����ĳ�෭ת
	bool have(int flipInd);  // �ж�ĳ�෭ת�Ƿ���Խ���
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
	Frame frame;  // ��¼��һ�η�ת��Ϣ���ͱ�״̬��waiting time�Լ�����

	void initialize();  // ��ʼ�����б��������״μ���frame
	void step();  // ����frame�еĽ������״̬���������µ�frame
	double getEnergy();  // ����ÿ�β������¼���������Ӧ����û���ۻ������ۼӷ�ʽ����ÿ��һ��ʱ��͵��ô˺�����ȷ����
private:
	const int nbrCNum;
	const int spinCNum;
	FlipCount* _flipCountMap;  // ÿ��ԭ�ӵ�������ת����
	int* _flipCountPerClass;  // ÿ����ת���Ͷ�Ӧ������
	double* _Pflips;  // ÿ����ת���Ͷ�Ӧ�ĸ���
	int* _candAtomInds;  // ����suggest������,��ź�ѡ��תԭ������.���ⷴ�������ڴ�
	double _deltaE; // �´ε������仯
	double _cdeltaE;  // Kahan����еĵ�λ����

	void initializeSigma();
	void initializeDeltaHs();
	void initializeFlipCounts();
	void initializePflips();
	void updateFCandPflips();  // ����frame����flipCountMap��Pflips
	void updateEnergy();

	int getEqualNbrNum(int n, int m);
	int getEqualNbrNum(int n, int m, int spin);
	void calAtomFlipCount(int n, int m);
	void calAtomFlipCount(int n, int m, int spin);
	Neighbors getNeighbors(int n, int m);

	void suggest(); // �ɸ��ʾ��������һ�η�ת�����������
	double calWaitingTime();
};

int choiceProb(double* prob, int size, random_engine& rng);
int choice(int* choiceList, int size, random_engine& rng);

pair<int*, Frame*> simulate(int steps, int N, int M, int q, double J, double h, double T, double tau, int randomSeed, int recordFreq, bool silent);  // ������, ���س�̬��ģ��켣
pair<int*, Frame*> simulate(int steps, int N, int M, int q, double J, double h, double T, double tau, int randomSeed, Frame* traj, int recordFreq, bool silent);  // ������, ���س�̬��ģ��켣
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
