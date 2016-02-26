#ifndef AISYSTEM_H
#define AISYSTEM_H

#include "cvUtils.h"

#include <ctime>
#include <iostream>

const unsigned FrameWidth = 480;
const unsigned FrameHeight = 480;

class AISystem
{
public:
	enum class Loop : unsigned
	{
		None = 0x00000,
		WaitForBackground = 0x10000,
		WaitForPerson = 0x20000,
		WaitForAction = 0x40000,
		WaitForAccept = 0x80000
	};

public:
	AISystem();

	~AISystem();

	void setSkinRatio(double skinRatio);

	void setEnergyRatio(double energyRatio);

	void setBinaryThreshold(uchar binaryThreshold);

	void setEnergyThreshold(uchar energyThreshold);

	void run();

	void updateMotionEnergy();

protected:
	// INFO
	Loop m_loop;
	unsigned long m_frame;

	// PARAMETERS
	double m_skinRatio;
	double m_energyRatio;

	uchar m_binaryThreshold;
	uchar m_energyThreshold;

	// RGB MATS
	cv::Mat m_currentRGB;
	cv::Mat m_photoRGB;

	// GREYSCALE MATS
	cv::Mat m_backgroundGREY;
	cv::Mat m_currentGREY;
	cv::Mat m_previousGREY;
	cv::Mat m_frameDifferenceGREY;
	cv::Mat m_backgroundDifferenceGREY;
	cv::Mat m_sumEnergyGREY;

	// BINARY MATS
	cv::Mat m_frameDifferenceBINARY;
	cv::Mat m_backgroundDifferenceBINARY;
	cv::Mat m_energyBINARY;

	cv::Mat m_energyProjXBINARY;
	cv::Mat m_energyProjYBINARY;

	// PROJECTION VECTORS
	std::vector<int> m_energyProjX;
	std::vector<int> m_energyProjY;

	// OPENCV
	cv::VideoCapture m_videoCapture;

	// GUI 
	cv::Mat m_exitScene;
	cv::Mat m_acceptRejectAction;
	cv::Mat m_comeIn;
	cv::Mat m_waitForAction;
};


#endif // AISYSTEM_H