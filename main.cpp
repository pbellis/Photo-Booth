#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "cvUtils.h"
#include "AISystem.h"

int main() 
{
	AISystem aisystem;

	aisystem.setBinaryThreshold(50);
	aisystem.setEnergyThreshold(240);
	aisystem.setEnergyRatio(1e-3);
	aisystem.setSkinRatio(0.1);

	aisystem.run();

	system("pause");

	return 0;
}