#include <mlpack/core.hpp>
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/RLNNCognitiveEngine.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/UDPServer.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/ViaSatDriver.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/EthUDPTransmitterSimple.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/ML605Driver.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/Logging.hpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/ASRPDriver.cpp"
#include <boost/date_time/posix_time/posix_time.hpp>

#include <fstream>

#include <chrono>
#include <thread>

#include <iostream>

using namespace mlpack;

int findIdxInActionList(const std::vector<int> &idxs, arma::Mat<int> & actionListIdxs) {
	int actionID = -1;
	bool skipFlag = false;

	for(int i=0; i<actionListIdxs.n_cols; i++) {
		skipFlag = false;
		for(int j=0; j<idxs.size() && skipFlag==false; j++) {
			if(actionListIdxs(j,i) != idxs[j]) {
				skipFlag = true;
			} else {
				//keep going
			}
		}
		if(skipFlag==false) {
			actionID = i;
			break;
		}
	}
	return actionID;
}

void doNothing(const boost::system::error_code&) {};

std::ofstream logFile("logging.txt",std::ofstream::out);
int main() {
	#ifdef LOGGING
	logFile << boost::posix_time::microsec_clock::local_time() << std::endl;
	logFile << "::Start of Log File" << std::endl;
	logFile << "-----------------" << std::endl;
	#endif

	//vars
	const int RSSI_UDP_PORT = 52008;
	const int FRAME_UDP_PORT = 50001;

	const std::vector<unsigned char> ETHTX_DEST_MAC_ADDR = {0x00,0x0A,0x35,0x02,0x5B,0x74};
	const std::vector<unsigned char> ETHTX_SRC_MAC_ADDR = {0x08,0x00,0x27,0x06,0xea,0x4e};
	const unsigned int ETHTX_UDP_SRC_PORT = 50002;
	const unsigned int ETHTX_UDP_DEST_PORT = 50002;
	const long ETHTX_TX_INTERVAL_MSEC = 1000;

	const bool continueFromFile = false;
	const bool saveToFile = true;
	const std::string rlnnLoadFilename = "emergency";
	const std::string rlnnSaveFilename = "save_test1";
	const bool SIMULATION_FLAG = true;
	const bool USE_SNR_PROFILE = true;
	const bool TX_POWER_PATCH_EN = true;
	
	int actionID;
	int lastActionID;
	arma::mat actionList;
	arma::Mat<int> actionListIdxs;
	std::vector<int> actionIDElements;
	actionIDElements.resize(4);
	std::vector<double> _snrProfileVec;

	//---------------------------------------------------//
	//Setting up Params
	//---------------------------------------------------//
	//input parameters
	std::cout<<"Setting Cog Engine Parameters" << std::endl;

	arma::rowvec fitnessWeights = {0.1, 0.8, 0.025,
									0.025, 0.025, 0.025};
	CogEngParams cogEngParams;
	//RLNNCognitiveEngine Params
	cogEngParams.cogeng_epsilonResetLim = 4e-3;
	cogEngParams.cogeng_nnExploreMaxPerfThresh = 0.9;
	cogEngParams.cogeng_nnRejectionRate = 0.95;
	cogEngParams.cogeng_trainFrac = 0.9;
	cogEngParams.cogeng_pruneFrac = 0.75;
	cogEngParams.cogeng_fitnessWeights = fitnessWeights;
	cogEngParams.cogeng_forceExploreThreshold = 0.95;

	//NeuralNetworkPredictor Params
	cogEngParams.nnExplore_nNets = 1;
	cogEngParams.nnExplore_inputVectorSize = 7;
	cogEngParams.nnExplore_hiddenLayerSizes.push_back(7);
	cogEngParams.nnExplore_hiddenLayerSizes.push_back(50);	
	cogEngParams.nnExplore_outputVectorSize = 1;
	cogEngParams.nnExplore_rmsProp_stepSize = 0.01;
	cogEngParams.nnExplore_rmsProp_alpha = 0.88;
	cogEngParams.nnExplore_rmsProp_eps = 1e-8;
	cogEngParams.nnExplore_rmsProp_maxEpochs = 100;
	cogEngParams.nnExplore_rmsProp_tolerance = 1e-18;
	cogEngParams.nnExplore_rmsProp_shuffle = true;
	cogEngParams.nnExploit_nNets = 1;
	cogEngParams.nnExploit_inputVectorSize = 7;
	cogEngParams.nnExploit_hiddenLayerSizes.push_back(20);	
	cogEngParams.nnExploit_outputVectorSize = 1;
	cogEngParams.nnExploit_rmsProp_stepSize = 0.01;
	cogEngParams.nnExploit_rmsProp_alpha = 0.88;
	cogEngParams.nnExploit_rmsProp_eps = 1e-8;
	cogEngParams.nnExploit_rmsProp_maxEpochs = 100;
	cogEngParams.nnExploit_rmsProp_tolerance = 1e-18;
	cogEngParams.nnExploit_rmsProp_shuffle = true;
	//Application Specific Object Params
	cogEngParams.nnAppSpec_nOutVecFeatures = 8;
	cogEngParams.nnAppSpec_frameSize = 16200.0;
	cogEngParams.nnAppSpec_maxEsN0 = 40.0;//12.93; //dB
	cogEngParams.nnAppSpec_minEsN0 = -20.0;//12.93; //dB
	cogEngParams.nnAppSpec_maxBER = pow(10,-6);
	cogEngParams.nnAppSpec_modList <<4<<4<<4<<4<<4<<4<<4<<4<<4
								  <<4<<8<<8<<8<<8<<8
								  <<16<<16<<16<<16<<16
								  <<32<<32<<32<<32
								  <<arma::endr;
	cogEngParams.nnAppSpec_codList <<(1.0/4.0)<<(1.0/3.0)<<(2.0/5.0)<<(1.0/2.0)
									<<(3.0/5.0)<<(2.0/3.0)<<(3.0/4.0)<<(4.0/5.0)
									<<(5.0/6.0)<<(8.0/9.0)
									<<(3.0/5.0)<<(2.0/3.0)<<(3.0/4.0)<<(5.0/6.0)
									<<(8.0/9.0)
								  	<<(2.0/3.0)<<(3.0/4.0)<<(4.0/5.0)<<(5.0/6.0)
								  	<<(8.0/9.0)
								  	<<(3.0/4.0)<<(4.0/5.0)<<(5.0/6.0)<<(8.0/9.0)
								  	<<arma::endr;
	cogEngParams.nnAppSpec_modCodList << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 9 << 10
									  << 12 << 13 << 14 << 15 << 16 
									  << 18 << 19 << 20 << 21 << 22
									  << 24 << 25 << 26 << 27 << arma::endr;
	cogEngParams.nnAppSpec_rollOffList << 0.35 << 0.25 << 0.20 <<arma::endr;
	double RsMin = 0.5*pow(10,6)/(1+cogEngParams.nnAppSpec_rollOffList.min());//0.5
	double RsMax = 5*pow(10,6)/(1+cogEngParams.nnAppSpec_rollOffList.max());
	//cogEngParams.nnAppSpec_symbolRateList = arma::trans(arma::regspace(RsMin,0.1*pow(10,6),RsMax)); //0.1MHz spacing
	cogEngParams.nnAppSpec_symbolRateList <<(1.0*pow(10,6))<<arma::endr;
	cogEngParams.nnAppSpec_transmitPowerList << -7.5 << -7.0 << -6.5 << -6.0
											 << -5.5 << -5.0 << -4.5 << -4.0
											 << -3.5 << -3.0 << -2.5 << -2.0
											 << -1.5 << -1.0 << -0.5 <<  0.0
											 << arma::endr;
	//cogEngParams.nnAppSpec_transmitPowerList << -0.01 << 0.0 << arma::endr;
	//cogEngParams.nnAppSpec_transmitPowerList = arma::trans(arma::regspace(-10.0,1.0,0.0)); //1.0 dB spacing
	//TrainingDataBuffer Params
	cogEngParams.buf_nTrainTestSamples = 200;

	#ifdef LOGGING
	logFile << boost::posix_time::microsec_clock::local_time() << std::endl;
	logFile << "::cogEngParams:" << std::endl;
	logFile << "::cogEngParams.cogeng_epsilonResetLim: " << cogEngParams.cogeng_epsilonResetLim << std::endl;
	logFile << "::cogEngParams.cogeng_nnExploreMaxPerfThresh: " << cogEngParams.cogeng_nnExploreMaxPerfThresh << std::endl;
	logFile << "::cogEngParams.cogeng_nnRejectionRate: " << cogEngParams.cogeng_nnRejectionRate << std::endl;
	logFile << "::cogEngParams.cogeng_fitnessWeights: " << cogEngParams.cogeng_fitnessWeights << std::endl;
	logFile << "::cogEngParams.cogeng_trainFrac: " << cogEngParams.cogeng_trainFrac << std::endl;
	logFile << "::cogEngParams.cogeng_pruneFrac: " << cogEngParams.cogeng_pruneFrac << std::endl;
	logFile << "::cogEngParams.cogeng_forceExploreThreshold: " << cogEngParams.cogeng_forceExploreThreshold << std::endl;
	//NeuralNetworkPredictor Params
	logFile << "::cogEngParams.nnExplore_nNets: " << cogEngParams.nnExplore_nNets << std::endl;
	logFile << "::cogEngParams.nnExplore_inputVectorSize: " << cogEngParams.nnExplore_inputVectorSize << std::endl;
	logFile << "::cogEngParams.nnExplore_hiddenLayerSizes: ";	
	for(int j=0; j<cogEngParams.nnExplore_hiddenLayerSizes.size(); j++) {
		logFile << cogEngParams.nnExplore_hiddenLayerSizes[j] << '\t';
	}
	logFile << std::endl;
	logFile << "::cogEngParams.nnExplore_outputVectorSize: " << cogEngParams.nnExplore_outputVectorSize << std::endl;
	logFile << "::cogEngParams.nnExploit_nNets: " << cogEngParams.nnExploit_nNets << std::endl;
	logFile << "::cogEngParams.nnExploit_inputVectorSize: " << cogEngParams.nnExploit_inputVectorSize << std::endl;
	logFile << "::cogEngParams.nnExploit_hiddenLayerSizes: ";
	for(int j=0; j<cogEngParams.nnExploit_hiddenLayerSizes.size(); j++) {
		logFile << cogEngParams.nnExploit_hiddenLayerSizes[j] << '\t';
	}
	logFile << std::endl;
	logFile << "::cogEngParams.nnExploit_outputVectorSize: " << cogEngParams.nnExploit_outputVectorSize << std::endl;
	//RMSProp params
	logFile << "::cogEngParams.nnExplore_rmsProp_stepSize: " << cogEngParams.nnExplore_rmsProp_stepSize << std::endl;
	logFile << "::cogEngParams.nnExplore_rmsProp_alpha: " << cogEngParams.nnExplore_rmsProp_alpha << std::endl;
	logFile << "::cogEngParams.nnExplore_rmsProp_eps: " << cogEngParams.nnExplore_rmsProp_eps << std::endl;
	logFile << "::cogEngParams.nnExplore_rmsProp_maxEpochs: " << cogEngParams.nnExplore_rmsProp_maxEpochs << std::endl;
	logFile << "::cogEngParams.nnExplore_rmsProp_tolerance: " << cogEngParams.nnExplore_rmsProp_tolerance << std::endl;
	logFile << "::cogEngParams.nnExplore_rmsProp_shuffle: " << cogEngParams.nnExplore_rmsProp_shuffle << std::endl;
	logFile << "::cogEngParams.nnExploit_rmsProp_stepSize: " << cogEngParams.nnExploit_rmsProp_stepSize << std::endl;
	logFile << "::cogEngParams.nnExploit_rmsProp_alpha: " << cogEngParams.nnExploit_rmsProp_alpha << std::endl;
	logFile << "::cogEngParams.nnExploit_rmsProp_eps: " << cogEngParams.nnExploit_rmsProp_eps << std::endl;
	logFile << "::cogEngParams.nnExploit_rmsProp_maxEpochs: " << cogEngParams.nnExploit_rmsProp_maxEpochs << std::endl;
	logFile << "::cogEngParams.nnExploit_rmsProp_tolerance: " << cogEngParams.nnExploit_rmsProp_tolerance << std::endl;
	logFile << "::cogEngParams.nnExploit_rmsProp_shuffle: " << cogEngParams.nnExploit_rmsProp_shuffle << std::endl;
	//App Specific Params
	logFile << "::cogEngParams.nnAppSpec_nOutVecFeatures: " << cogEngParams.nnAppSpec_nOutVecFeatures << std::endl;
	logFile << "::cogEngParams.nnAppSpec_frameSize: " << cogEngParams.nnAppSpec_frameSize << std::endl;
	logFile << "::cogEngParams.nnAppSpec_maxEsN0: " << cogEngParams.nnAppSpec_maxEsN0 << std::endl;
	logFile << "::cogEngParams.nnAppSpec_maxBER: " << cogEngParams.nnAppSpec_maxBER << std::endl;
	logFile << "::cogEngParams.nnAppSpec_modList: " << cogEngParams.nnAppSpec_modList << std::endl;
	logFile << "::cogEngParams.nnAppSpec_codList: " << cogEngParams.nnAppSpec_codList << std::endl;
	logFile << "::cogEngParams.nnAppSpec_rollOffList: " << cogEngParams.nnAppSpec_rollOffList << std::endl;
	logFile << "::cogEngParams.nnAppSpec_symbolRateList: " << cogEngParams.nnAppSpec_symbolRateList << std::endl;
	logFile << "::cogEngParams.nnAppSpec_transmitPowerList: " << cogEngParams.nnAppSpec_transmitPowerList << std::endl;
	logFile << "::cogEngParams.nnAppSpec_modCodList: " << cogEngParams.nnAppSpec_modCodList << std::endl;
	//TrainingDataBuffer Params
	logFile << "::cogEngParams.buf_nTrainTestSamples: " << cogEngParams.buf_nTrainTestSamples << std::endl;
	logFile << "-----------------" << std::endl;

	logFile << "::RLNNTester Params:" << std::endl;
	logFile << "::continueFromFile: " <<  continueFromFile << std::endl;
	logFile << "::saveToFile: " << saveToFile << std::endl;
	logFile << "::rlnnLoadFilename: " << rlnnLoadFilename << std::endl;
	logFile << "::rlnnSaveFilename: " << rlnnSaveFilename << std::endl;
	logFile << "::SIMULATION_FLAG: " << SIMULATION_FLAG << std::endl;
	logFile << "::TX_POWER_PATCH_EN: " << TX_POWER_PATCH_EN << std::endl;

	#endif

	actionList.set_size(6,cogEngParams.nnAppSpec_symbolRateList.n_elem
							*cogEngParams.nnAppSpec_transmitPowerList.n_elem
							*cogEngParams.nnAppSpec_modList.n_elem
							*cogEngParams.nnAppSpec_rollOffList.n_elem);
	actionListIdxs.set_size(4,cogEngParams.nnAppSpec_symbolRateList.n_elem
							*cogEngParams.nnAppSpec_transmitPowerList.n_elem
							*cogEngParams.nnAppSpec_modList.n_elem
							*cogEngParams.nnAppSpec_rollOffList.n_elem);
	int id = 0;
	for(int i1=0; i1<cogEngParams.nnAppSpec_symbolRateList.n_elem; i1++) {
		for(int i2=0; i2<cogEngParams.nnAppSpec_transmitPowerList.n_elem; i2++) {
			for(int i3=0; i3<cogEngParams.nnAppSpec_modList.n_elem; i3++) {
				for(int i4=0; i4<cogEngParams.nnAppSpec_rollOffList.n_elem; i4++) {
						actionList(0,id) = cogEngParams.nnAppSpec_symbolRateList(i1);
						actionListIdxs(0,id) = i1;
						actionList(1,id) = cogEngParams.nnAppSpec_transmitPowerList(i2);
						actionListIdxs(1,id) = i2;
						actionList(2,id) = (double) cogEngParams.nnAppSpec_modCodList(i3); //modcod
						actionListIdxs(2,id) = i3;
						actionList(3,id) = cogEngParams.nnAppSpec_rollOffList(i4);
						actionListIdxs(3,id) = i4;
						id++;
				}
			}
		}
	}
	for(int i=0; i<actionList.n_cols; i++) {
		actionList(4,i) = (double) cogEngParams.nnAppSpec_modList(actionListIdxs(2,i));
		actionList(5,i) = cogEngParams.nnAppSpec_codList(actionListIdxs(2,i));
		actionList(2,i) = log2(actionList(4,i));
	}

	#ifdef LOGGING
	logFile << boost::posix_time::microsec_clock::local_time() << std::endl;
	logFile << "::Action List:" << std::endl;
	for(int i=0; i<actionList.n_cols; i++) {
		for(int j=0; j<actionList.n_rows; j++) {
			logFile << actionList(j,i) << "\t";
		}
		logFile << std::endl;
	}
	logFile << "-----------------" << std::endl;
	#endif

	#ifdef LOGGING
	logFile << boost::posix_time::microsec_clock::local_time() << std::endl;
	logFile << "::Action Idxs:" << std::endl;
	for(int i=0; i<actionListIdxs.n_cols; i++) {
		for(int j=0; j<actionListIdxs.n_rows; j++) {
			logFile << actionListIdxs(j,i) << "\t";
		}
		logFile << std::endl;
	}
	logFile << "-----------------" << std::endl;
	#endif


	cogEngParams.buf_actionList = actionList;

	//---------------------------------------------------//
	//Instantiations and Set-up
	//---------------------------------------------------//
	//instantiate cog engine
	std::cout<<"Instantiating Cog Engine" << std::endl;
	RLNNCognitiveEngine rlnnCogEng(cogEngParams);
	if(continueFromFile) {
		rlnnCogEng.loadOldRun(rlnnLoadFilename);
	}

	//instantiate UDP TLM Receiver
	ViaSatDriver vsDrvr;
	ViaSatDriver frameDrvr;
	ViaSatDriver::RSSIMessageFormat rssiMsg;
	ViaSatDriver::FrameMessageFormat frameMsg;
	int udp_status;

	//instantiate UDP transmitter
	ML605Driver ml605;
	ML605Driver::ML605MessageFormat ml605Msg;
	std::vector<unsigned char> ml605Buf;

	//ASRPDriver asrp("192.168.50.108:80");

	//init socket to listen for EsN0/RSSI on port 13
	boost::asio::io_service io_service;
	//boost::asio::io_service::work work(io_service);

	UDPServer rssiServer(io_service,RSSI_UDP_PORT);
	UDPServer frameServer(io_service,FRAME_UDP_PORT);
	std::vector<unsigned char> * rssiBuff;
	std::vector<unsigned char> * frameBuff;
	int packetSize;

	//init socket to send to ML605
	EthUDPTransmitterSimple ethTx(io_service, ETHTX_UDP_DEST_PORT, ETHTX_UDP_SRC_PORT,
			ETHTX_DEST_MAC_ADDR, ETHTX_SRC_MAC_ADDR);

	//open sockets and continuously listen
	std::thread tRSSI([&io_service](){ io_service.run(); });
	std::thread tFrames([&io_service](){ io_service.run(); });
	
	//set up way to stop execution cleanly
	bool quitExecution = false;
	std::thread tTimer([&quitExecution]() 
	{
		while(std::cin.get() != 'q');
		quitExecution = true;
	});


	//load in SNR Profile (if enabled)
	if(USE_SNR_PROFILE) {
		std::string line;
		std::ifstream snrProfileFile("snrProfile.txt");
		while(std::getline(snrProfileFile,line)) {
			std::istringstream ss(line);
			std::string token;

			while(std::getline(ss,token)) {
				_snrProfileVec.push_back(std::atof(token.c_str()));
			}
		}
		snrProfileFile.close();
	}

	//---------------------------------------------------//
	//Main Execution
	//---------------------------------------------------//
	//choose an action
	arma::rowvec measurementVec(6);
	double rcvdEsN0;
	int i=0;
	for(int i=0; (i<10400) && (!quitExecution) ; i++) {
		#ifdef LOGGING
		logFile << "::Iteration: " << i << std::endl;
		logFile << "::Start Time: " << boost::posix_time::microsec_clock::local_time() << std::endl;
		#endif

		//ACTION CHOOSING--------------------------------//
		//choose action
		//TODO: make this it's own thread
		std::cout<<i<<": Choosing Action"<<std::endl;
		actionID = rlnnCogEng.chooseAction();

		if(TX_POWER_PATCH_EN) {
			actionIDElements[0] = actionListIdxs(0,actionID);
			actionIDElements[2] = actionListIdxs(2,actionID);
			actionIDElements[3] = actionListIdxs(3,actionID);						
			if(i != 0) {
				if(std::abs(actionList(1,actionID)-actionList(1,lastActionID)) > 1.5) { //if tx pwr change is more than 1.5 dB
					if((actionList(1,actionID)-actionList(1,lastActionID)) > 1.5) { //new power is >1.5 dB larger
						//find action with pwr 1.5 dB larger
						actionIDElements[1] = actionListIdxs(1,lastActionID)+3; //each idx is 0.5 dB steps. 3*0.5 dB = 1.5 dB						
						actionID = findIdxInActionList(actionIDElements,actionListIdxs);

					} else { //new power is >1.5 dB smaller
						//find action with pwr 1.5 dB smaller
						actionIDElements[1] = actionListIdxs(1,lastActionID)-3; //each idx is 0.5 dB steps. 3*0.5 dB = 1.5 dB					
						actionID = findIdxInActionList(actionIDElements,actionListIdxs);
					}
				}
					
			} else {
				if(std::abs(0.0 - actionList(1,actionID)) > 1.5) { //power starts at 0.00 dB by default
					//find action with pwr 1.5 dB smaller
					actionIDElements[1] = actionListIdxs.row(1).max()-3; //each idx is 0.5 dB steps. 3*0.5 dB = 1.5 dB					
					actionID = findIdxInActionList(actionIDElements,actionListIdxs);				
				}
			}
			lastActionID = actionID;
		}

		std::cout<<i<<": Action ID Chosen: " << actionID << std::endl;
		std::cout<<i<<": Symbol Rate: " << actionList(0,actionID)  << " Symbols/Sec" << std::endl;
		std::cout<<i<<": Relative TX Power: " << actionList(1,actionID) << " dB" << std::endl;
		std::cout<<i<<": Filter Roll Off: " << actionList(3,actionID) << std::endl;
		std::cout<<i<<": Modulation: " << cogEngParams.nnAppSpec_modList(actionListIdxs(2,actionID)) << std::endl;
		std::cout<<i<<": Code Rate: " << cogEngParams.nnAppSpec_codList(actionListIdxs(2,actionID)) << std::endl;
		std::cout<<i<< ": ModCod Idx: " << cogEngParams.nnAppSpec_modCodList(actionListIdxs(2,actionID)) << std::endl;		

		
		#ifdef LOGGING
		logFile << "::Action Chosen: " << boost::posix_time::microsec_clock::local_time() << std::endl;
		logFile << "::Action ID: " << actionID << std::endl;
		logFile << "::Action Params: ";
		for(int i=0; i<actionList.n_rows; i++) {
			logFile << actionList(i,actionID) << "\t";
		}
		logFile << std::endl;
		#endif

		//send action
		std::cout<<i<<": Transmitting Action"<<std::endl;
		ml605Msg.modcod = cogEngParams.nnAppSpec_modCodList(actionListIdxs(2,actionID));
		ml605Msg.rolloff = actionListIdxs(3,actionID);
		ml605Msg.transmitPower = actionListIdxs(1,actionID);
		//ml605Msg.transmitPower = 15;
		ml605Msg.enable = 1;
		ml605.generateTXActionMessage(ml605Msg,ml605Buf);
		ethTx.updateUDPPayload(ml605Buf);
		ethTx.sendFrame();
		//asrp.sendAction(cogEngParams.nnAppSpec_modCodList(actionListIdxs(2,actionID)), actionList(1,actionID), actionList(3,actionID));

		//RECORDING RESPONSE-----------------------------//
		//std::cout << "Send next UDP packet to be received." << std::endl;
		#ifdef LOGGING
		logFile << "::Action Sent: " << boost::posix_time::microsec_clock::local_time() << std::endl;
		#endif

		//std::cin.get();
		//block until worst case RTT has passed
		boost::asio::deadline_timer t(io_service,boost::posix_time::milliseconds(40));
		t.wait();

		#ifdef LOGGING
		logFile << "::Action Received: " << boost::posix_time::microsec_clock::local_time() << std::endl;
		#endif

		if(!SIMULATION_FLAG) {
			//read response from rssi packet
			std::cout<<i<<": Reading Received Action" << std::endl;
			rssiBuff = rssiServer.getRecvBuffer();
			packetSize = rssiServer.getPacketSize();
			udp_status = vsDrvr.getRSSIPacketContents(rssiBuff,rssiMsg,packetSize);
			if(rssiMsg.rcvLock) {
				rcvdEsN0 = rssiMsg.esN0;
			}
			std::cout<<i<<": Receive Lock: " << rssiMsg.rcvLock << std::endl;
			#ifdef LOGGING
				logFile << "::Receive Lock: " << rssiMsg.rcvLock << std::endl;
			#endif
			//read response from frames
			//udp_status = 0;
			//while(!udp_status) { //we don't want to block because frames might be missing
				//frameBuff = frameServer.getRecvBuffer();
				//packetSize = frameServer.getPacketSize();
				//udp_status = frameDrvr.getFramePacketContents(frameBuff,frameMsg,packetSize);
			//}
			//populate measurement vector
			measurementVec(0) = rcvdEsN0;
			measurementVec(1) = cogEngParams.nnAppSpec_transmitPowerList(actionListIdxs(1,actionID)); //tx power
			measurementVec(2) = cogEngParams.nnAppSpec_symbolRateList(actionListIdxs(0,actionID)); //Rs
			measurementVec(3) = cogEngParams.nnAppSpec_rollOffList(actionListIdxs(3,actionID)); //rolloff
			measurementVec(4) = cogEngParams.nnAppSpec_modList(actionListIdxs(2,actionID)); //mod
			measurementVec(5) = cogEngParams.nnAppSpec_codList(actionListIdxs(2,actionID)); //cod
		} else {
			//measurementVec(0) = 6.0+math::Random()-0.5 + cogEngParams.nnAppSpec_transmitPowerList(ml605Msg.transmitPower);
			double SNR;
			if(USE_SNR_PROFILE) {
				SNR = _snrProfileVec[i];
			} else {
				if(i<6000) {//6000
					SNR=((double)i)*0.002*2;
				} else {
					SNR=12.0-0.002*(i-6000)*2;
				}
			}
			double noise = mlpack::math::RandNormal(0,0.01); //std deviation ~0.1 dB

			measurementVec(0) = SNR + noise + cogEngParams.nnAppSpec_transmitPowerList(actionListIdxs(1,actionID));
			measurementVec(1) = cogEngParams.nnAppSpec_transmitPowerList(actionListIdxs(1,actionID)); //tx power
			measurementVec(2) = cogEngParams.nnAppSpec_symbolRateList(actionListIdxs(0,actionID)); //Rs
			measurementVec(3) = cogEngParams.nnAppSpec_rollOffList(actionListIdxs(3,actionID)); //rolloff
			measurementVec(4) = cogEngParams.nnAppSpec_modList(actionListIdxs(2,actionID)); //mod
			measurementVec(5) = cogEngParams.nnAppSpec_codList(actionListIdxs(2,actionID)); //cod
		}
		//we don't need to measure log2(M) since we have M.
		std::cout <<i<<": EsN0 measured: " << measurementVec(0) << " dB" << std::endl;

		#ifdef LOGGING
		logFile << "::Measurement Received:";
		logFile << measurementVec;
		#endif

		//record response of environment
		std::cout<<i<<": Recording Response" << std::endl;
		std::cout<<"---------------------------" << std::endl;
/*		if(!SIMULATION_FLAG) {
			actionIDElements[0] = actionListIdxs(0,actionID);
			actionIDElements[1] = frameMsg.txPower;
			actionIDElements[2] = frameMsg.modCod;
			actionIDElements[3] = frameMsg.rollOff;
		} else {
			actionIDElements[0] = actionListIdxs(0,actionID);
			actionIDElements[1] = ml605Msg.transmitPower;
			actionIDElements[2] = ml605Msg.modcod;
			actionIDElements[3] = ml605Msg.rolloff;			
		}
		actionID = findIdxInActionList(actionIDElements,actionListIdxs);
*/
		rlnnCogEng.recordResponse(actionID, measurementVec);
		#ifdef LOGGING
		logFile << "::End Time: " << boost::posix_time::microsec_clock::local_time() << std::endl;
		logFile << std::endl;
		#endif

		std::cout << std::endl << std::endl;	
	}
	rssiServer.close();
	frameServer.close();
	tRSSI.join();
	tFrames.join();
	tTimer.join();

	logFile.close();
	if(saveToFile) {
		rlnnCogEng.saveCurrentRun(rlnnSaveFilename);
	}
	return 0;
}