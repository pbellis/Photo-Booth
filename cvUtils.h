#ifndef CVUTILS_H
#define CVUTILS_H

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//skin detection
void mySkinDetect(Mat& src, Mat& dst);

//Background differencing
void myBackgroundDifferencing(Mat& src, Mat& dst, Mat& bg);

//And function between 2 binary matrices
void myBinaryAnd(Mat& src1, Mat& src2, Mat& dst);
int myMin(int a, int b, int c);
int myMax(int a, int b, int c);

/** Creates a grayscale image from a color image.
@param src The source color image
@param dst The destination grayscale image
*/
void myGrayScale(Mat& src, Mat& dst);
void getEnergyProjX(Mat& energyImg, vector<int>& v);
void getEnergyProjY(Mat& energyImg, vector<int>& v);

/**
Creates a tinted image from a color image.
@param src The source color image
@param dst The destination tinted image
@param channel The channel specifies the tint
*/
void myTintImage(Mat& src, Mat& dst, int channel);
/**

*/
void myDiffImage(Mat& frame1, Mat& frame2, Mat& dst);

/**
Creates a thresholded image from a grayscale image.

@param src The source color image
@param dst The destination tinted image
@param threshold The specified threshold intensity
*/
void myThresholdImage(Mat& src, Mat& dst, int threshold);

/**
Calculates the motion energy over the last (255-zeroThreshold) frames.
The energy image pixels are either 0 if no motion was detected in the frame window
or a value in the interval (zeroThreshold, 255] depending on how much time has passed since 
the energy was detected at that pixel (lower values are older)
*/
void myMotionEnergy(Mat& diff, Mat& energy);

void processFrame(const Mat &src, Mat &dst, int it = 2);

double getWhiteRatio(const Mat &src);

unsigned getWhitePixels(const Mat &src);

void energyProjectionMat(const vector<int> &x, const vector<int> &y, Mat &xDst, Mat &yDst);

void tEnergyProjection(const Mat &sumEnergy, vector<int> &xProj, vector<int> &yProj, int t = 255, float r = 0.15);

void tSmooth(const vector<int> &xProj, const vector<int> &yProj, vector<int> &xSmooth, vector<int> &ySmooth, int it = 2);

bool rejectReaction(vector<int>& xProj, vector<int>& yProj, double ratio = 0.2, double stdYOverXRatio = 2.3);

bool acceptReaction(vector<int>& xProj, vector<int>& yProj, double ratio = 0.2, double stdXOverYRatio = 3.0);

void computeBinaryCentroid(const Mat &src, Point2d &centroid);

void computeProjectionCentroid(const vector<int> &proj, Point2d &centroid);

void computeProjectionCorrelation(const vector<int> &xProj, const vector<int> &yProj, double xMean, double yMean, double &correlation);

bool waveReaction(const vector<int> &xProj, const vector<int> &yProj, double hRatio = 0.4, double aRatio = 0.4, double cRatio = 0.7);

bool isOkSign(Mat& src, Mat &webcam);

#endif
