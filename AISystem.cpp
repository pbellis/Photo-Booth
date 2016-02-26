#include "AISystem.h"

// HELPT COMMIT THIS

AISystem::AISystem()
	: m_videoCapture(0)
{
	
	m_videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, FrameWidth);
	m_videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, FrameHeight);

	// RGB
	m_currentRGB = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC3);
	m_photoRGB = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC3);

	// GREY
	m_backgroundGREY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);
	m_currentGREY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);
	m_previousGREY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);
	m_frameDifferenceGREY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);
	m_backgroundDifferenceGREY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);
	m_sumEnergyGREY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);

	// BINARY
	m_frameDifferenceBINARY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);
	m_backgroundDifferenceBINARY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);
	m_energyBINARY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);

	m_energyProjXBINARY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);
	m_energyProjYBINARY = cv::Mat::zeros(FrameHeight, FrameWidth, CV_8UC1);

	m_exitScene = imread("images/exit.png");
	m_acceptRejectAction = imread("images/photoAction.png");
	m_comeIn = imread("images/come.png");
	m_waitForAction = imread("images/action.png");

	m_energyProjX.resize(FrameWidth);
	m_energyProjY.resize(FrameHeight);

	m_frame = 0;
}

AISystem::~AISystem()
{
}

void AISystem::setSkinRatio(double skinRatio)
{
	m_skinRatio = skinRatio;
}

void AISystem::setEnergyRatio(double energyRatio)
{
	m_energyRatio = energyRatio;
}

void AISystem::setBinaryThreshold(uchar binaryThreshold)
{
	m_binaryThreshold = binaryThreshold;
}

void AISystem::setEnergyThreshold(uchar energyThreshold)
{
	m_energyThreshold = energyThreshold;
}

void AISystem::run()
{
	clock_t start, end;
	vector<int> tX, tY;

	m_loop = Loop::WaitForBackground;

	do
	{
		start = clock();

		m_videoCapture.read(m_currentRGB);
		Size size(FrameWidth, FrameHeight);
		resize(m_currentRGB, m_currentRGB, size);

		cv::imshow("Live", m_currentRGB);

		myGrayScale(m_currentRGB, m_currentGREY);
		cv::imshow("Live (Greyscale)", m_currentGREY);
		
		updateMotionEnergy();
		
		switch (m_loop)
		{
		case Loop::WaitForBackground:
			imshow("GUI", m_exitScene);

			if (getWhiteRatio(m_energyBINARY) < m_energyRatio && m_frame > 30)
			{
				m_backgroundGREY = m_currentGREY.clone();
				m_loop = Loop::WaitForPerson;
			}
			break;
		case Loop::WaitForPerson:
			imshow("GUI", m_comeIn);

			if (getWhiteRatio(m_energyBINARY) > m_energyRatio) {
				m_loop = Loop::WaitForAction;
			}
			break;
		case Loop::WaitForAction:
			imshow("GUI", m_waitForAction);

			myBackgroundDifferencing(m_currentGREY, m_backgroundDifferenceGREY, m_backgroundGREY);
			myThresholdImage(m_backgroundDifferenceGREY, m_backgroundDifferenceBINARY, 15);
			cv::imshow("Live (Background Difference Binary)", m_backgroundDifferenceBINARY);

			tEnergyProjection(m_energyBINARY, tX, tY, 255, 0.4f);
			tSmooth(tX, tY, m_energyProjX, m_energyProjY, 3);

			energyProjectionMat(m_energyProjX, m_energyProjY, m_energyProjXBINARY, m_energyProjYBINARY);
			cv::imshow("Live (X Motion Energy)", m_energyProjXBINARY);
			cv::imshow("Live (Y Motion Energy)", m_energyProjYBINARY);

			m_photoRGB = m_currentRGB.clone();

			if (isOkSign(m_backgroundDifferenceBINARY, m_photoRGB)) {
				m_loop = Loop::WaitForAccept;
			}
			else if (waveReaction(m_energyProjX, m_energyProjY, 0.2, 0.09, 0.85))
			{
				m_loop = Loop::None;
			}

			break;
		case Loop::WaitForAccept:
			imshow("GUI", m_acceptRejectAction);

			getEnergyProjX(m_energyBINARY, m_energyProjX);
			getEnergyProjY(m_energyBINARY, m_energyProjY);

			if (acceptReaction(m_energyProjX, m_energyProjY)) {
				imwrite("photo.png", m_photoRGB);
				cout << endl << endl << "Your photo has been saved as photo.png" << endl << "Thank you!" << endl;
				m_loop = Loop::None;
			}
			else if (rejectReaction(m_energyProjX, m_energyProjY)) {
				destroyWindow("photo");
				m_loop = Loop::WaitForAction;
			}

			break;
		}

		m_previousGREY = m_currentGREY.clone();

		m_frame++;


		if (cv::waitKey(10) >= 0) {
			break;
		}

		end = clock();

#define CASE(X) \
	case(X): \
		std::cout << "["#X"] "<< (end - start) * 1000 / CLOCKS_PER_SEC << "ms" << endl; \
		break;

		switch (m_loop)
		{
			//CASE(Loop::None)
			CASE(Loop::WaitForAction)
			CASE(Loop::WaitForBackground)
			CASE(Loop::WaitForPerson)
			CASE(Loop::WaitForAccept)
		};

#undef CASE
	} while (m_loop != Loop::None);

	destroyAllWindows();
	m_videoCapture.release();
}

void AISystem::updateMotionEnergy() {
	myDiffImage(m_currentGREY, m_previousGREY, m_frameDifferenceGREY);
	myThresholdImage(m_frameDifferenceGREY, m_frameDifferenceBINARY, m_binaryThreshold);
	cv::imshow("Live (Frame Difference)", m_frameDifferenceBINARY);

	myMotionEnergy(m_frameDifferenceBINARY, m_sumEnergyGREY);
	myThresholdImage(m_sumEnergyGREY, m_energyBINARY, m_energyThreshold);
	cv::imshow("Live (Motion Energy)", m_energyBINARY);
}
