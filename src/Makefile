CPPC = g++
CCARGS = -O3 -DNDEBUG --std=c++14
CCARGS_DEBUG = -O0 -g -Wall --std=c++14
SOURCE = potts.cpp main.cpp
INCLUDES = -I. -I ~/include/tclap-1.2.5/include 

PottsKMC: $(SOURCE)
	$(CPPC) $(CCARGS) -o $@ $(INCLUDES) $^

.PHONY: debug
debug: PottsKMC_g
PottsKMC_g: $(SOURCE)
	$(CPPC) $(CCARGS_DEBUG) -o $@ $(INCLUDES) $^

.PHONY: clean
clean:
	@rm -f PottsKMC PottsKMC_g
