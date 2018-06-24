#include <vector>

class ML605Driver {
public:
	ML605Driver() {}

	struct ML605MessageFormat {
		unsigned char modcod;
		unsigned char rolloff;
		unsigned char transmitPower;
		unsigned char enable;
	};

	void generateTXActionMessage(ML605MessageFormat msg, std::vector <unsigned char> &ethMsg) {
		ethMsg.resize(4);

		ethMsg[0] = 0x80; //defined as this.
		ethMsg[0] = 0x80 | ((0x0F - msg.transmitPower) & 0x0F); //defined as this, transmit power has largest value when most negative.
		ethMsg[1] = msg.modcod & 0x1F;
		//ethMsg[2] = ((msg.transmitPower & 0x0F)<<2) | (msg.enable & 0x01);
		ethMsg[2] = msg.enable & 0x01;
		ethMsg[3] = msg.rolloff & 0x03;
	}

private:
};