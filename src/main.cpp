#include "potts.h"

const char* artKMC = 
"########################################################################################################\n"
" ________  ________  _________  _________  ________           ___  __    _____ ______   ________     \n"
"|\\   __  \\|\\   __  \\|\\___   ___\\\\___   ___\\\\   ____\\         |\\  \\|\\  \\ |\\   _ \\  _   \\|\\   ____\\    \n"
"\\ \\  \\|\\  \\ \\  \\|\\  \\|___ \\  \\_\\|___ \\  \\_\\ \\  \\___|_        \\ \\  \\/  /|\\ \\  \\\\\\__\\ \\  \\ \\  \\___|    \n"
" \\ \\   ____\\ \\  \\\\\\  \\   \\ \\  \\     \\ \\  \\ \\ \\_____  \\        \\ \\   ___  \\ \\  \\\\|__| \\  \\ \\  \\       \n"
"  \\ \\  \\___|\\ \\  \\\\\\  \\   \\ \\  \\     \\ \\  \\ \\|____|\\  \\        \\ \\  \\\\ \\  \\ \\  \\    \\ \\  \\ \\  \\____  \n"
"   \\ \\__\\    \\ \\_______\\   \\ \\__\\     \\ \\__\\  ____\\_\\  \\        \\ \\__\\\\ \\__\\ \\__\\    \\ \\__\\ \\_______\\\n"
"    \\|__|     \\|_______|    \\|__|      \\|__| |\\_________\\        \\|__| \\|__|\\|__|     \\|__|\\|_______|\n"
"                                             \\|_________|                                            \n"
"                                                                                                     \n"
"########################################################################################################\n"
"Author: 潘高翔 北京大学化学与分子工程学院\n";

pair<int*, Frame*> simulate(int steps, int N, int M, int q, double J, double h, double T, double tau, int randomSeed, int recordFreq, bool silent)
{
	Frame* traj = new Frame[steps / recordFreq + 1];
	return simulate(steps, N, M, q, J, h, T, tau, randomSeed, traj, recordFreq, silent);
}

pair<int*, Frame*> simulate(int steps, int N, int M, int q, double J, double h, double T, double tau, int randomSeed, Frame* traj, int recordFreq, bool silent)
{
	PottsModel model(N, M, q, J, h, T, tau, randomSeed);
	int* initState = new int[N * M];
	Frame* curFrame = traj;
	model.initialize();
	copy(model.sigma, model.sigma + model.modelSize, initState);
	*(curFrame++) = model.frame;
	for (int i = 1; i < steps + 1; ++i) {
		model.step();
		if (i % recordFreq == 0) {
			*(curFrame++) = model.frame;
			if (!silent)
				printf("step %d/%d, %f%%, Energy = %e\n", i, steps, (double)i * 100 / steps, model.frame.E);
		}
	}
	return pair<int*, Frame*>(initState, traj);
}

void simulate(int steps, int N, int M, int q, double J, double h, double T, double tau, int randomSeed, ostream& _initStr, ostream& _traj, int recordFreq, bool silent)
{
	PottsModel model(N, M, q, J, h, T, tau, randomSeed);
	model.initialize();
	writeState(_initStr, model.sigma, N, M);
	if (!silent)
		writeState(cout, model.sigma, N, M);
	if (recordFreq == 1) {
		writeTrajHead(_traj, OnlyChange, model.sigma, N, M);
		_traj << model.frame;
		for (int i = 1; i < steps + 1; ++i) {
			model.step();
			
			_traj << model.frame;
			if (!silent)
				printf("step %d/%d, %f%%, Energy = %e\n", i, steps, (double)i * 100 / steps, model.frame.E);
		}
	}
	else {
		writeTrajHead(_traj, FullState, model.sigma, N, M);
		_traj << 0 << ' ';
		writeState(_traj, model.sigma, N * M);
		_traj << model.frame.dt << ' ' << model.frame.E << '\n';
		for (int i = 1; i < steps + 1; ++i) {
			model.step();
			if (i % recordFreq == 0) {
				_traj << i << ' ';
				writeState(_traj, model.sigma, N * M);
				_traj << model.frame.dt << ' ' << model.frame.E << '\n';
				if (!silent)
					printf("step %d/%d, %f%%, Energy = %e\n", i, steps, (double)i * 100 / steps, model.frame.E);
			}
		}
	}
}

//int main() {
//	int N = 10, M = 10, q = 2, steps = 200, recordFreq = 1;
//	double J = 1., T = 1., tau = 1., h = 0.;
//	// pair<int*, Frame*> res = simulate(steps, N, M, q, J, h, T, tau, 1234, recordFreq, false);
//
//	// ofstream initFile("potts.init");
//	auto& initFile = cout;
//	// ofstream trajFile("potts.traj");
//	auto& trajFile = cout;
//	simulate(steps, N, M, q, J, h, T, tau, 1234, cout, cout, recordFreq, true);
//	return 0;
//}

void parse_args(int& N, int& M, int& q, int& steps, int& recordFreq, int& randomSeed, double& J, double& T, double& tau, double& h, string& jobName, bool& silent, int argc, char* argv[])
{
	try {
		TCLAP::CmdLine cmd("KMC simulator for Potts model\nAuthor: 潘高翔，北京大学化学与分子工程学院", ' ', _VERSION, true);
		TCLAP::ValueArg<int> N_("N", "nrows", "", true, 0, "int", cmd);
		TCLAP::ValueArg<int> M_("M", "ncols", "if not defined, take N as its value", false, -1, "int", cmd);
		TCLAP::ValueArg<int> q_("q", "spin-classes", "", true, 0, "int", cmd);
		TCLAP::ValueArg<int> steps_("", "steps", "simulation steps", true, 0, "int", cmd);
		TCLAP::ValueArg<int> recordFreq_("", "record-freq", "record traj every how many steps", false, 1, "int", cmd);
		TCLAP::ValueArg<int> randomSeed_("", "random-seed", "", false, rand(), "int", cmd);
		TCLAP::ValueArg<double> J_("J", "coupling-coef", "coupling coefficient", true, 1., "float", cmd);
		TCLAP::ValueArg<double> h_("B", "magnetic-coef", "", true, 0., "float", cmd);
		TCLAP::ValueArg<double> T_("T", "temperature", "kB * temprature", true, 1., "float", cmd);
		TCLAP::ValueArg<double> tau_("", "tau", "time-scaling factor", false, 10., "float", cmd);
		TCLAP::ValueArg<string> jobName_("", "job-name", "prefix of output files", false, "potts", "string", cmd);
		TCLAP::SwitchArg silent_("", "quiet", "", cmd, false);
		cmd.parse(argc, argv);

		N = N_.getValue();
		M = M_.getValue(); M = (M >= 0 ? M : N);
		q = q_.getValue();
		steps = steps_.getValue();
		recordFreq = recordFreq_.getValue();
		randomSeed = randomSeed_.getValue();
		J = J_.getValue();
		h = h_.getValue();
		T = T_.getValue();
		tau = tau_.getValue();
		jobName = jobName_.getValue();
		silent = silent_.getValue();
	}
	catch (const exception& e) {
		cout << e.what();
		abort();
	}
}

int main(int argc, char* argv[]) {
	int N, M, q = 2, steps, recordFreq, randomSeed;
	double J, T, tau, h;
	string jobName;
	bool silent;
	parse_args(N, M, q, steps, recordFreq, randomSeed, J, T, tau, h, jobName, silent, argc, argv);

	ofstream initStateStr(jobName + ".init");
	ofstream trajStr(jobName + ".traj");
	trajStr << setprecision(15);
	ofstream logStr(jobName + ".log");

	time_t rawtime;
	struct tm timeinfo;
	char buffer[60];
	time(&rawtime);
#ifdef _MSVC_LANG
	localtime_s(&timeinfo, &rawtime);
#else
	localtime_r(&rawtime, &timeinfo);
#endif // _MSVC_LANG
	strftime(buffer, 60, "%Y-%m-%d %H:%M:%S", &timeinfo);
	logStr << artKMC
		<< buffer << "\n\n"
		<< "PARAMS\n"
		<< "N = " << N << '\n'
		<< "M = " << M << '\n'
		<< "q = " << q << '\n'
		<< "steps = " << steps << '\n'
		<< "recordFreq = " << recordFreq << '\n'
		<< "randomSeed = " << randomSeed << '\n'
		<< "J = " << J << '\n'
		<< "B = " << h << '\n'
		<< "T = " << T << '\n'
		<< "tau = " << tau << '\n'
		<< "jobName = " << jobName << "\n\n\n";

	time_t startTime = time(NULL);
	simulate(steps, N, M, q, J, h, T, tau, randomSeed, initStateStr, trajStr, recordFreq, silent);
	time_t endTime = time(NULL);
	logStr << "Simulation process cost " << difftime(endTime, startTime) << " seconds\n";
}
