#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/UDPServer.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/ViaSatDriver.cpp"
#include <chrono>
#include <thread>

int main()
{
	const int RSSI_UDP_PORT = 52005;
	const int FRAME_UDP_PORT = 50001;

	//instantiate UDP TLM Receiver
	ViaSatDriver vsDrvr;
	ViaSatDriver frameDrvr;
	ViaSatDriver::RSSIMessageFormat rssiMsg;
	ViaSatDriver::FrameMessageFormat frameMsg;
	int udp_status;

	//init socket to listen for EsN0/RSSI on port 13
	boost::asio::io_service io_service;
	UDPServer rssiServer(io_service,RSSI_UDP_PORT);
	UDPServer frameServer(io_service,FRAME_UDP_PORT);
	std::vector<char> * rssiBuff;
	std::vector<char> * frameBuff;
	int packetSize;

	//open sockets and continuously listen
	std::thread tRSSI([&io_service](){ io_service.run(); });
	std::thread tFrames([&io_service](){ io_service.run(); });


	while(true)
	{
		//read response from rssi packet
		std::cout << "Grabbing Most Recent EsN0" << std::dec << std::endl;
		rssiBuff = rssiServer.getRecvBuffer();
		packetSize = rssiServer.getPacketSize();
		udp_status = vsDrvr.getRSSIPacketContents(rssiBuff,rssiMsg,packetSize);

		if(!udp_status) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		//read response from frames
		/*std::cout << "Waiting for Next Frame" << std::dec << std::endl;
		udp_status=0;
		while(!udp_status) {
			frameBuff = frameServer.getRecvBuffer();
			packetSize = frameServer.getPacketSize();
			udp_status = frameDrvr.getFramePacketContents(frameBuff,frameMsg,packetSize);
		}
		std::cout << "Received Next Frame" << std::dec << std::endl;*/

		//print new packet to console
		std::cout << "Radio ID: " << std::hex << rssiMsg.radioID << std::endl;
		std::cout << "Time Stamp (sec): " << std::hex << rssiMsg.tStampSec << std::endl;
		std::cout << "Time Stamp (usec):: " << std::hex << rssiMsg.tStampUSec << std::endl;
		std::cout << "Receiver Lock: " << std::hex << rssiMsg.rcvLock << std::endl;
		std::cout << "RX Power: " << std::dec << rssiMsg.rxPower << std::endl;
		std::cout << "EsN0: " << std::dec << rssiMsg.esN0 << std::endl;
		std::cout << std::endl;
		std::cout << "Frame Number: " << std::dec << (unsigned int) frameMsg.frameNumber << std::endl;
		std::cout << "TX Power: " << std::dec << (unsigned int) frameMsg.txPower << std::endl;
		std::cout << "Roll Off: " << std::dec << (unsigned int) frameMsg.rollOff << std::endl;
		std::cout << "ModCod: " << std::dec << (unsigned int) frameMsg.modCod << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;

	}
	rssiServer.close();
	frameServer.close();
	tRSSI.join();
	tFrames.join();
}