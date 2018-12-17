CPPFLAGS += -std=gnu++11
CPPFLAGS += -larmadillo
CPPFLAGS += -lmlpack
CPPFLAGS += -lboost_serialization
CPPFLAGS += -lboost_system
CPPFLAGS += -lpthread
CPPFLAGS += -lcurl
CPPFLAGS += -g


lm :	CPPFLAGS += -DMAKEFILE_MEM
lm :	CPPFLAGS += -DLM=true
lm :	CPPFLAGS += -DRLM=false
lm :	CPPFLAGS += -DNSE=false

rlm :	CPPFLAGS += -DMAKEFILE_MEM
rlm :	CPPFLAGS += -DLM=false
rlm :	CPPFLAGS += -DRLM=true
rlm :	CPPFLAGS += -DNSE=false

nse :	CPPFLAGS += -DMAKEFILE_MEM
nse :	CPPFLAGS += -DLM=false
nse :	CPPFLAGS += -DRLM=false
nse :	CPPFLAGS += -DNSE=true	

default : RLNNCognitiveEngineTester_v3.cpp
	g++ -o RLNNCognitiveEngineTester_v3 RLNNCognitiveEngineTester_v3.cpp $(CPPFLAGS) 2> err.txt

v2 : RLNNCognitiveEngineTester_v2.cpp
	g++ -o RLNNCognitiveEngineTester_v2 RLNNCognitiveEngineTester_v2.cpp -std=gnu++11  -larmadillo -lmlpack -lboost_serialization -lboost_system -pthread -lcurl -O2

all: lm rlm nse

lm : RLNNCognitiveEngineTester_v3.cpp
	g++ -o RLNNCognitiveEngineTester_v3_lm RLNNCognitiveEngineTester_v3.cpp $(CPPFLAGS) 2> err.txt
rlm : RLNNCognitiveEngineTester_v3.cpp
	g++ -o RLNNCognitiveEngineTester_v3_rlm RLNNCognitiveEngineTester_v3.cpp $(CPPFLAGS) 2> err.txt
nse : RLNNCognitiveEngineTester_v3.cpp
	g++ -o RLNNCognitiveEngineTester_v3_nse RLNNCognitiveEngineTester_v3.cpp $(CPPFLAGS) 2> err.txt


clean:
	rm RLNNCognitiveEngineTester_v3
	rm RLNNCognitiveEngineTester_v3_lm
	rm RLNNCognitiveEngineTester_v3_rlm
	rm RLNNCognitiveEngineTester_v3_nse
