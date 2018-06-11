#include <string>
#include <iostream>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdio.h>

//NOTE: REQURIES APPLICATION CALLED "curl"
//in Linux repository.  There's also a windows port.
//https://curl.haxx.se/download.html
//Make sure you can call curl from the command line.

class ASRPDriver {
public:

	void sendAction(int modcod, double txPower, double rolloff);

	ASRPDriver(const std::string &ipAddressWithPortNum) {
		getHandlerID(ipAddressWithPortNum);
	}

private:
	std::string _strsHandler;
	std::string _ipAddressWithPort;

	void getHandlerID(const std::string &ipAddressWithPortNum);
	void strsConfigure(const std::string &propID, const std::string &propData, bool quotesNeeded);
	void strsQuery(const std::string &propID, std::string &propData);

	std::string execCommand(const std::string cmd);

	void sendModCod(int modcod);
	void sendRollOff(double rolloff);
	void sendTXPower(double txPower);

};

std::string ASRPDriver::execCommand(const std::string cmd) {
	const char * cmd_c = cmd.c_str();

	char buffer[128];
	std::string result = "";
	FILE* pipe = popen(cmd_c,"r");
	if(!pipe) throw std::runtime_error("popen() failed!");
	try {
		while(!feof(pipe)) {
			if(fgets(buffer,128,pipe) != NULL) {
				result += buffer;
			}
		}
	} catch (...) {
		pclose(pipe);
		std::cout << "ERROR!!!" << std::endl;
		throw;
	}
	pclose(pipe);
	return result;
}

void ASRPDriver::getHandlerID(const std::string &ipAddressWithPortNum) {
	std::string part_1 = "curl -s --ipv4 --data-binary \'\"DVBS2\"\' ";
	std::string part_3 = "/STRS_HandleRequest.fcgi";
	_strsHandler = execCommand(part_1 + ipAddressWithPortNum + part_3);
	_ipAddressWithPort = ipAddressWithPortNum;
}

void ASRPDriver::strsConfigure(const std::string &propID, const std::string &propData, bool quotesNeeded) {
	std::string part_1;
	std::string part_3;
	std::string part_5;
	std::string part_7;

	part_1 = "curl -s --ipv4 --data-binary \'{\"toWF\":12648460,\"propID\":\"";
	if(quotesNeeded) {
		part_3 = "\",\"propData\":\"";
		part_5 = "\"}\' ";
	} else {
		part_3 = "\",\"propData\":";
		part_5 = "}\' ";		
	}
	part_7 = "/STRS_Configure.fcgi";

	std::string cmd = part_1 + propID + part_3 + propData + part_5 + _ipAddressWithPort + part_7;
	execCommand(cmd);
}

/*void ASRPDriver::strsQuery(const std::string &propID, std::string &propData) {
	std::string part_1 = "curl -s --ipv4 --data-binary \'{\"toWF\":12648460,\"propID\":\"";
	std::string part_3 = "\",\"propData\":\"";
	std::string part_5 = "\"}\' ";
	std::string part_7 = "/STRS_Query.fcgi";

	std::string cmd = part_1 + propID + part_3 + "1" + part_5 + _ipAddressWithPort + part_7;
	propData = execCommand(cmd);
}*/

void ASRPDriver::sendModCod(int modcod) {
	std::string modcodStr = "modcod";
	std::string modcodVal = std::to_string(modcod);
	strsConfigure(modcodStr,modcodVal,true);
}

void ASRPDriver::sendTXPower(double txPower) {
	std::string txPowerStr = "dac_iq_scaler";
	std::string txPowerVal = std::to_string(txPower);
	strsConfigure(txPowerStr,txPowerVal.substr(0,txPowerVal.length()-3-1),false);
}

void ASRPDriver::sendRollOff(double rolloff) {
	std::string rolloffStr = "rolloff";
	std::string rolloffVal = std::to_string(rolloff);
	strsConfigure(rolloffStr,rolloffVal.substr(0,rolloffVal.length()-3-1),true);
}

void ASRPDriver::sendAction(int modcod, double txPower, double rolloff) {
	sendModCod(modcod);
	sendTXPower(txPower);
	sendRollOff(rolloff);
}
/*
int main() {
	ASRPDriver asrp("http://192.168.50.108:80");
	asrp.sendAction(10,-15.0,0.20);
}*/