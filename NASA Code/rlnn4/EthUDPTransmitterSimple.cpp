#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <net/ethernet.h>
#include <sys/socket.h>
#include <linux/if_packet.h>
#include <cstddef>
#include <thread>

using boost::asio::ip::udp;

typedef boost::asio::generic::raw_protocol raw_protocol_t;
typedef boost::asio::generic::basic_endpoint<raw_protocol_t> raw_endpoint_t;

class EthUDPTransmitterSimple {
public:
	EthUDPTransmitterSimple(boost::asio::io_service& io_service,
					  unsigned int destPortNumber, unsigned int srcPortNumber,
					  const std::vector<unsigned char> & destMACAddr, 
					  const std::vector<unsigned char> & srcMACAddr)
		: socket_(io_service,raw_protocol_t(PF_PACKET, SOCK_RAW)),
		  io_service_(io_service)
		  {
			//update MAC addrs
			for(int i=0; i<destMACAddr.size(); i++) {
				data_[0+i] = destMACAddr[i];
			}
			for(int i=0; i<srcMACAddr.size(); i++) {
				data_[6+i] = srcMACAddr[i];
			}

			//update port numbers
			data_[34] = (unsigned char) (srcPortNumber>>8);
			data_[35] = (unsigned char) (srcPortNumber&0x00FF);
			data_[36] = (unsigned char) (destPortNumber>>8);
			data_[37] = (unsigned char) (destPortNumber&0x00FF);

			memset(&sockaddr_,0,sizeof(sockaddr));
			sockaddr_.sll_family = PF_PACKET;
			sockaddr_.sll_protocol = htons(ETH_P_ALL);
			sockaddr_.sll_ifindex = if_nametoindex(ifname.c_str());
			sockaddr_.sll_hatype = 1;

			socket_.bind(raw_endpoint_t(&sockaddr_, sizeof(sockaddr_)));
			
		  }

	void updateUDPPayload(const std::vector<unsigned char> &data) {
		//resize to new packet size
		data_.resize(46+data.size()); //46 bytes of header/CRC
		
		//add new data into packet
		for(int i=0; i<data.size(); i++) {
			data_[i+42]=data[i];
		}

		//add udp checksum placeholder
		data_[data_.size()-1] = 0x00;
		data_[data_.size()-2] = 0x00;
		data_[data_.size()-3] = 0x00;
		data_[data_.size()-4] = 0x00;
	}

	//void close()
	//{
	//	continue_sending_ = false;
		//io_service closes after it doesn't have any work left
	//}


	void sendFrame()
	{
		socket_.send(boost::asio::buffer(data_));
	}

private:
	bool continue_sending_;
	std::string ifname = "enp0s3";
	sockaddr_ll sockaddr_;
	boost::asio::io_service& io_service_;
	raw_protocol_t::socket socket_;
	std::vector<unsigned char> data_ =
		{0x00,0x0A,0x35,0x02,0x5B,0x74, //dest MAC address "ML605!"
		 0x08,0x00,0x27,0x06,0xea,0x4e, //src MAC address
		 0x08,0x00, 					//ipV4 ethertype
		 0x45,0x00,						//ipv4 type, ip header length default DSCP, no ECN
		 0x00,0x20,						//ip packet length
		 0x00,0x00,0x00,0x00,			//ID, flags, fragment offset
		 0x00,0x11,0x00,0x00,			//TTL,Protocol,Header Checksum
		 0xC0,0xA8,0x01,0x60,			//Source IP Address (192.168.1.96)
		 0xC0,0xA8,0x01,0x61,			//Dest IP Address (192.168.1.97)
		 0xC3,0x52,0xC3,0x52,			//UDP source and dest ports (50002)
		 0x00,0x0C,						//UDP packet size 8+4=12
		 0x00,0x00,						//UDP Checksum
		 0xFF,0xFF,0xFF,0xFF,			//UDP PAYLOAD DATA
		 0x00,0x00,0x00,0x00			//Ethernet CRC
		};


};

/*int main() {
	const std::vector<unsigned char> DEST_MAC_ADDR = {0x00,0x0A,0x35,0x02,0x5B,0x74};
	const std::vector<unsigned char> SRC_MAC_ADDR = {0x08,0x00,0x27,0x06,0xea,0x4e};
	const unsigned int UDP_SRC_PORT = 50002;
	const unsigned int UDP_DEST_PORT = 50002;
	const long TX_INTERVAL_MSEC = 40;

	boost::asio::io_service io_service;

	EthUDPTransmitter ethTx(io_service, UDP_DEST_PORT, UDP_SRC_PORT,
							DEST_MAC_ADDR, SRC_MAC_ADDR, TX_INTERVAL_MSEC);
	std::thread t([&io_service](){ io_service.run(); });

	std::cout << "Press Return to Quit" << std::endl;
	std::cin.get();
	ethTx.close();
	t.join();

	return 0;
}*/