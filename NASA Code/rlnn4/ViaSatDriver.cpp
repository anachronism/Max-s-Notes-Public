#include <vector>
#include <mlpack/core.hpp>
#include <cstdint>

class ViaSatDriver {
public:
	ViaSatDriver() {
		lastRSSITStampSec_ = 0;
		lastRSSITStampUSec_ = 0;

		lastFrameNumber_ = 255;
	}

	struct RSSIMessageFormat{
		uint32_t radioID;
		uint32_t tStampSec;
		uint32_t tStampUSec;
		bool rcvLock;
		double rxPower;
		double esN0;
	};

	struct FrameMessageFormat{
		unsigned char frameNumber;
		unsigned char txPower;
		unsigned char rollOff;
		unsigned char modCod;
	};

	int getRSSIPacketContents(std::vector<unsigned char>* buff, RSSIMessageFormat &rssiMsg, int packetSize) {
		uint16_t rssiInt;
		uint16_t esN0Int;
		if(packetSize==16) {

			rssiMsg.radioID = ((uint32_t)((*buff)[0]))<<24 | ((uint32_t)((*buff)[1]))<<16 | ((uint32_t)((*buff)[2]))<<8 | ((uint32_t)((*buff)[3]));
			rssiMsg.tStampSec = (*buff)[4]<<24 | (*buff)[5]<<16 | (*buff)[6]<<8 | (*buff)[7];
			rssiMsg.tStampUSec = (*buff)[8]<<24 | (*buff)[9]<<16 | (*buff)[10]<<8 | (*buff)[11];
			rssiMsg.rcvLock = (bool) (((*buff)[12]&0x80)>>7);
			
			rssiInt = ((*buff)[12]&0x7F)<<8 | (*buff)[13];
			rssiMsg.rxPower = ((double)rssiInt)/(-10.0);

			esN0Int = (*buff)[14]<<8 | (*buff)[15];
			rssiMsg.esN0 = ((double)esN0Int)/10.0 - 5.0;

			if(rssiMsg.tStampSec > lastRSSITStampSec_ || ((rssiMsg.tStampSec==lastRSSITStampSec_) && (rssiMsg.tStampUSec>lastRSSITStampUSec_))) {
				lastRSSITStampSec_ = rssiMsg.tStampSec;
				lastRSSITStampUSec_ = rssiMsg.tStampUSec;
				return 1; //you got a new packet
			} else {
				return 0; //you already have requested this packet
			}
		} else {
			return 0;
		}
	}


	int getFramePacketContents(std::vector<unsigned char>* buff, FrameMessageFormat &frameMsg, int packetSize) {
		unsigned char modCod;
		if(packetSize==16) { //16200/8
			frameMsg.frameNumber = (unsigned char) ((*buff)[0]);
			frameMsg.txPower = (unsigned char) (((*buff)[1]&0xF0)>>4);
			frameMsg.rollOff = (unsigned char) (((*buff)[1]&0x0C)>>2);
			frameMsg.modCod = (unsigned char) ((((*buff)[1]&0x03)>>0)<<3 | ((*buff)[2]&0xE0)>>5 );

			if(frameMsg.frameNumber != lastFrameNumber_) {
				lastFrameNumber_ = frameMsg.frameNumber;
				return 1; //you got a new packet
			} else {
				return 0; //you already have requested this packet
			}
		} else {
			return 0;
		}
	}


private:
	uint32_t lastRSSITStampSec_;
	uint32_t lastRSSITStampUSec_;

	unsigned char lastFrameNumber_;
};
