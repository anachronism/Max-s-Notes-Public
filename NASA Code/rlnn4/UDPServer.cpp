//hello_world_server.cpp

#include <iostream>
#include <thread>
#include <string>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <vector>

using boost::asio::ip::udp;

class UDPServer
{
public:
	UDPServer(boost::asio::io_service& io_service, int portNumber)
		: socket_(io_service, udp::endpoint(udp::v4(),portNumber)),
		  io_service_(io_service)
		{
			recv_buffer_.resize(65536);

			start_receive();
		}

	void close()
	{
		//posts to thread that is running io_service that it needs to
		//close the socket.
		io_service_.post([this]() { socket_.close(); });
	}

	std::vector<unsigned char>* getRecvBuffer() {
		return &recv_buffer_;
	}

	size_t getPacketSize() {
		return packet_size;
	}

private:
	void start_receive()
	{
		socket_.async_receive_from(
			boost::asio::buffer(recv_buffer_,recv_buffer_.size()), remote_endpoint_,
			boost::bind(&UDPServer::handle_receive, this,
				boost::asio::placeholders::error,
				boost::asio::placeholders::bytes_transferred));
	}

	void handle_receive(const boost::system::error_code& error,
		std::size_t bytes_transferred)
	{
		if(!error || error == boost::asio::error::message_size)
		{
			

			//std::cout << "Buffer: ";
			//for(int i=0; i< bytes_transferred; i++) {
			//	std::cout << recv_buffer_[i];
			//}
			//std::cout << std::endl;
			packet_size=bytes_transferred;
			start_receive();
		}

		//std::cout << "in receive handle" << std::endl;
	}

	boost::asio::io_service& io_service_;
	udp::socket socket_;
	udp::endpoint remote_endpoint_;
	std::vector<unsigned char> recv_buffer_;
	size_t packet_size;

};

/*int main()
{
	try
	{
		boost::asio::io_service io_service;
		udp_server server(io_service,13);
		std::thread t([&io_service](){ io_service.run(); });
		std::cout << "do i get here?" << std::endl;

		int cmd = 0;
		while(cmd != 1)
		{
			//std::cout << "hi";
			std::cin >> cmd;
		}
		std::cout << "left while loop" << std::endl;
		server.close();
		t.join();
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
*/