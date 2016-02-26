#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "cvUtils.h"
#include <opencv2/features2d.hpp>

#include <ctime>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

bool isOkSign(Mat& src, Mat &webcam) {
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.filterByColor = true;
	params.blobColor = 0;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 300;
	params.maxArea = 15000;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.4;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.7;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.6;

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	vector<KeyPoint> keypoints;
	detector->detect(src, keypoints);

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	if (keypoints.size() == 1) {
		Mat im_with_keypoints;
		drawKeypoints(webcam, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		
		webcam = im_with_keypoints;

		// Show blobs
		imshow("photo", im_with_keypoints);

		return true;
	} else {
		return false;
	}
}

//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
	return max(max(a, b), c);
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
	return min(min(a, b), c);
}

//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.

	dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		Vec3b* src_ptr = src.ptr<Vec3b>(i);
		uchar* dst_ptr = dst.ptr<uchar>(i);
			
		for (int j = 0; j < src.cols; j++) {
			Vec3b vals = src_ptr[j]; //BGR values

			uchar b = vals[0];
			uchar g = vals[1];
			uchar r = vals[2];

			bool isSkin = (r>95) && (b > 20) && (g > 40) && (myMax(r, g, b) - myMin(r, g, b)) > 15 && (abs(r - g) > 15) && r > g && r > b;

			if (isSkin) {
				dst_ptr[j] = 255;
			}
		}
	}

	//TODO:
	//Use the following test for skin color detection
	//Red > 95 and Blue > 20 and Green > 40, and 
	//max(Red, Green, Blue) - min(Red, Green, Blue) > 15, and
	//abs(Red - Green) > 15, and
	//Red > Green, and
	//Red > Blue
}

void getEnergyProjX(Mat& energyImg, vector<int>& v) {
	for (int i = 0; i < energyImg.cols; i++) {
		int e = 0;
		for (int j = 0; j < energyImg.rows; j++) {
			if (energyImg.at<uchar>(j, i) == 255) {
				e++;
			}
		}

		v[i] = e;
	}
}

void getEnergyProjY(Mat& energyImg, vector<int>& v) {
	for (int i = 0; i < energyImg.rows; i++) {
		int e = 0;

		for (int j = 0; j < energyImg.cols; j++) {
			if (energyImg.at<uchar>(i, j) == 255) {
				e++;
			}
		}

		v[i] = e;
	}
}

//Creates a grayscale image from a color image.
void myGrayScale(Mat& src, Mat& dst) {
	//Different algorithms for converting color to grayscale: http://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/
	Mat dst1 = Mat::zeros(src.rows, src.cols, CV_8UC1); //Returns a zero array of same size as src mat, and of type CV_8UC1
	for (int i = 0; i < src.rows; i++) {
		Vec3b* src_ptr = src.ptr<Vec3b>(i); //Color
		uchar* dst_ptr = dst1.ptr<uchar>(i); //Greyscale
		for (int j = 0; j < src.cols; j++) {
			//For each pixel, compute the average intensity of the 3 color channels
			Vec3b intensity = src_ptr[j]; //Vec3b is a vector of 3 uchar (unsigned character)
			int avg = (intensity[0] + intensity[1] + intensity[2]) / 3;
			dst_ptr[j] = avg;
		}
	}

	dst = dst1;
	//cvtColor(src, dst, CV_BGR2GRAY); //cvtColor documentation: http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html
}

//Creates a tinted image from a color image.
void myTintImage(Mat& src, Mat& dst, int channel)
{
	dst = src.clone(); //the clone methods creates a deep copy of the matrix
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			//For each pixel, suppress the channels other than that passed in the argument of the function
			dst.at<Vec3b>(i, j)[(channel + 1) % 3] = 0;
			dst.at<Vec3b>(i, j)[(channel + 2) % 3] = 0;
		}
	}
}

void myDiffImage(Mat & frame1, Mat & frame2, Mat & dst)
{
	for (int i = 0; i < frame1.rows; i++) {
		uchar* f1_ptr = frame1.ptr<uchar>(i);
		uchar* f2_ptr = frame2.ptr<uchar>(i);
		uchar* dst_ptr = dst.ptr<uchar>(i);
		for (int j = 0; j < frame1.cols; j++) {
			dst_ptr[j] = abs(f2_ptr[j] - f1_ptr[j]);
		}
	}
}

//Creates a thresholded image from a grayscale image.
void myThresholdImage(Mat& src, Mat& dst, int threshold)
{
	for (int i = 0; i < src.rows; i++) {
		uchar* src_ptr = src.ptr<uchar>(i);
		uchar* dst_ptr = dst.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) {
			//For each pixel, assign intensity value of 0 if below threshold, else assign intensity value of 255
			int intensity = src_ptr[j];
			if (intensity < threshold)	{
				dst_ptr[j] = 0;
			} else {
				dst_ptr[j] = 255;
			}
		}
	}
}


void myBackgroundDifferencing(Mat& src,Mat& dst, Mat& bg) {
	for (int i = 0; i < src.rows; i++) {
		uchar* src_ptr = src.ptr<uchar>(i);
		uchar* bg_ptr = bg.ptr<uchar>(i);
		uchar* dst_ptr = dst.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j) {
			dst_ptr[j] = abs(src_ptr[j] - bg_ptr[j]);
		}
	}
}
	
void myBinaryAnd(Mat& src1, Mat& src2, Mat& dst) {
	dst = Mat::zeros(src1.rows, src1.cols, CV_8UC1);
	for (int i = 0; i < src1.rows; i++) {
		uchar* s1_ptr = src1.ptr<uchar>(i);
		uchar* s2_ptr = src2.ptr<uchar>(i);
		uchar* dst_ptr = dst.ptr<uchar>(i);
		for (int j = 0; j < src1.cols; ++j) {
			if (s1_ptr[j] > 200 && s2_ptr[j] > 200) {
				dst_ptr[j] = 255;
			}
		}
	}
}


void myMotionEnergy(Mat& diff, Mat& energy) {
	for (int i = 0; i < diff.rows; i++) {
		uchar* diff_ptr = diff.ptr<uchar>(i);
		uchar* en_ptr = energy.ptr<uchar>(i);
		for (int j = 0; j < diff.cols; j++) {
			if (diff_ptr[j] > 200) {
				en_ptr[j] = 255;
			}
			else if (en_ptr[j] > 0) {
				en_ptr[j]--;
			}
		}
	}
}

// ADDED BY PATRICK

void processFrame(const Mat &src, Mat &dst, int it)
{
	unsigned i, j;

	Mat tmp = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
	const Vec3b *src_ptr;
	Vec3b *tmp_ptr;

	if (it < 1)
	{
		dst = src.clone();
		return;
	}

	for (i = 1; i < src.rows - 1; i++)
	{
		src_ptr = src.ptr<Vec3b>(i);
		tmp_ptr = tmp.ptr<Vec3b>(i);
		for (j = 1; j < src.cols - 1; j++)
		{
			tmp_ptr[j][0] = 0.25 * src_ptr[j - 1][0] + 0.5 * src_ptr[j][0] + 0.25 * src_ptr[j + 1][0];
			tmp_ptr[j][1] = 0.25 * src_ptr[j - 1][1] + 0.5 * src_ptr[j][1] + 0.25 * src_ptr[j + 1][1];
			tmp_ptr[j][2] = 0.25 * src_ptr[j - 1][2] + 0.5 * src_ptr[j][2] + 0.25 * src_ptr[j + 1][2];
		}
	}
	

	for (i = 0; i < src.rows; i++)
	{
		tmp_ptr = tmp.ptr<Vec3b>(i);
		for (j = 0; j < src.cols; j++)
		{
			tmp_ptr[j][0] = (130 + ((tmp_ptr[j][0] + 4) / 10) * 10) / 2;
			tmp_ptr[j][1] = (130 + ((tmp_ptr[j][1] + 4) / 10) * 10) / 2;
			tmp_ptr[j][2] = (130 + ((tmp_ptr[j][2] + 4) / 10) * 10) / 2;
		}
	}

	processFrame(tmp, dst, it - 1);
}

double getWhiteRatio(const Mat &src)
{
	unsigned i, j;
	const uchar *src_ptr;

	unsigned whiteCount = 0;
	const unsigned total = src.cols * src.rows;

	for (i = 0; i < src.rows; i++)
	{
		src_ptr = src.ptr<uchar>(i);
		for (j = 0; j < src.cols; j++)
		{
			if (src_ptr[j] > 200)
			{
				whiteCount++;
			}
		}
	}

	return (double(whiteCount) / total);
}

unsigned getWhitePixels(const Mat &src)
{
	unsigned i, j;
	const uchar *src_ptr;

	unsigned whiteCount = 0;

	for (i = 0; i < src.rows; i++)
	{
		src_ptr = src.ptr<uchar>(i);
		for (j = 0; j < src.cols; j++)
		{
			if (src_ptr[j] > 200)
			{
				whiteCount++;
			}
		}
	}

	return whiteCount;
}

void energyProjectionMat(const vector<int> &x, const vector<int> &y, Mat &xDst, Mat &yDst)
{
	unsigned i, j;
	uchar *xDst_ptr, *yDst_ptr;

	xDst = Mat::zeros(y.size(), x.size(), CV_8UC1);
	yDst = Mat::zeros(y.size(), x.size(), CV_8UC1);

	for (i = 0; i < x.size(); i++)
	{
		yDst_ptr = yDst.ptr<uchar>(i);
		xDst_ptr = xDst.ptr<uchar>(i);
		for (j = 0; j < y.size(); j++)
		{
			if (j < x[i]) xDst_ptr[j] = 255;
			if (i < y[j]) yDst_ptr[j] = 255;
		}
	}
}

void tEnergyProjection(const Mat &sumEnergy, vector<int> &xProj, vector<int> &yProj, int t, float r)
{
	unsigned i, j;
	const uchar *sumEnergy_ptr;

	xProj.resize(sumEnergy.cols);
	fill(xProj.begin(), xProj.end(), 0);

	yProj.resize(sumEnergy.rows);
	fill(yProj.begin(), yProj.end(), 0);

	for (i = 0; i < sumEnergy.rows; i++)
	{
		sumEnergy_ptr = sumEnergy.ptr<uchar>(i);
		for (j = 0; j < sumEnergy.cols; j++)
		{
			if (sumEnergy_ptr[j] == t) xProj[i] ++;
			if (sumEnergy_ptr[j] == t) yProj[j] ++;
		}
	}

	for (int &x : xProj)
	{
		x = x > r * sumEnergy.rows ? x : 0;
	}

	for (int &y : yProj)
	{
		y = y > r * sumEnergy.cols ? y : 0;
	}
}

void tSmooth(const vector<int> &xProj, const vector<int> &yProj, vector<int> &xSmooth, vector<int> &ySmooth, int it)
{
	unsigned i;
	vector<int> xtSmooth, ytSmooth;

	if (it < 1)
	{
		xSmooth = xProj;
		ySmooth = yProj;
		return;
	}

	xtSmooth.resize(xProj.size());
	fill(xtSmooth.begin(), xtSmooth.end(), 0);

	for (i = 1; i < xProj.size() - 1; i++)
	{
		xtSmooth[i] = (0.25 * xProj[i - 1]) + (0.5 * xProj[i]) + (0.25 * xProj[i + 1]);
	}

	ytSmooth.resize(yProj.size());
	fill(ytSmooth.begin(), ytSmooth.end(), 0);

	for (i = 1; i < yProj.size() - 1; i++)
	{
		ytSmooth[i] = (0.25 * yProj[i - 1]) + (0.5 * yProj[i]) + (0.25 * yProj[i + 1]);
	}

	tSmooth(xtSmooth, ytSmooth, xSmooth, ySmooth, it - 1);
}

bool acceptReaction(vector<int>& xProj, vector<int>& yProj, double ratio, double stdXOverYRatio)
{
	Scalar m, s;
	meanStdDev(xProj, m, s);
	double pixelMeanX = m[0];
	double pixelStdX = s[0];

	meanStdDev(yProj, m, s);
	double pixelMeanY = m[0];
	double pixelStdY = s[0];

	return (pixelMeanX / yProj.size() > ratio) && (pixelStdX > stdXOverYRatio * pixelStdY);
}

bool rejectReaction(vector<int>& xProj, vector<int>& yProj, double ratio, double stdYOverXRatio)
{
	Scalar m, s;
	meanStdDev(xProj, m, s);
	double pixelMeanX = m[0];
	double pixelStdX = s[0];

	meanStdDev(yProj, m, s);
	double pixelMeanY = m[0];
	double pixelStdY = s[0];

	cout << "Y axis mean: " << pixelMeanY << " Std: " << pixelStdY << endl;
	return (pixelMeanY / xProj.size() > ratio) && (pixelStdX * stdYOverXRatio < pixelStdY);
}

void computeBinaryCentroid(const Mat &src, Point2d &centroid)
{
	unsigned i, j;
	double n = 0;

	Point2d tmp(0,0);

	const uchar *src_ptr;

	for (i = 0; i < src.rows; i++)
	{
		src_ptr = src.ptr<uchar>(i);
		for (j = 0; j < src.cols; j++)
		{
			if (src_ptr[i] == 255)
			{
				tmp.x += j;
				tmp.y += i;
				n += 1.0;
			}
		}
	}

	if (n < 0.9)
	{
		centroid.x = 0;
		centroid.y = 0;
	}
	else
	{
		tmp.x /= n;
		tmp.y /= n;
		centroid = tmp;
	}
}

void computeProjectionCentroid(const vector<int> &proj, Point2d &centroid)
{
	unsigned i;
	double n = 0.0;

	Point2d tmp(0, 0);

	for (i = 0; i < proj.size(); i++)
	{
		if (proj[i] > 0)
		{
			tmp.x += i;
			tmp.y += proj[i];
			n += 1.0;
		}
	}

	if (n < 0.9)
	{
		centroid.x = 0;
		centroid.y = 0;
	}
	else
	{
		tmp.x /= n;
		tmp.y /= n;
		centroid = tmp;
	}
}

void computeProjectionCorrelation(const vector<int> &xProj, const vector<int> &yProj, double xMean, double yMean, double &correlation)
{
	double x;
	double y;

	double sumXY = 0.0;
	double sumXX = 0.0;
	double sumYY = 0.0;

	unsigned i;

	for (i = 0; i < xProj.size(); i++)
	{
		x = xProj[i] - xMean;
		y = yProj[i] - yMean;
		sumXY += x * y;
		sumXX += x * x;
		sumYY += y * y;
	}

	sumXX = sqrt(sumXX);
	sumYY = sqrt(sumYY);

	correlation = sumXY / (sumXX * sumYY);
}

bool waveReaction(const vector<int> &xProj, const vector<int> &yProj, double hRatio, double aRatio, double cRatio)
{
	Scalar m, s;
	meanStdDev(xProj, m, s);
	double xMean = m[0];
	double xDev = s[0];

	meanStdDev(yProj, m, s);
	double yMean = m[0];
	double yDev = s[0];

	double xArea = 0.0;
	double yArea = 0.0;

	for (int x : xProj) xArea += x;
	for (int y : yProj) yArea += y;

	double correlation;
	computeProjectionCorrelation(xProj, yProj, xMean, yMean, correlation);

	//std::cout << "----------------------------------------------------------------------" << std::endl;
	//std::cout << "correlation = " << abs(correlation) << std::endl;
	//std::cout << "xMean = " << xMean << ", needs to exceed " << hRatio * yProj.size() << std::endl;
	//std::cout << "yMean = " << yMean << ", needs to exceed " << hRatio * xProj.size() << std::endl;
	//std::cout << "xArea = " << xArea << ", needs to exceed " << aRatio * xProj.size() * yProj.size() << std::endl;
	//std::cout << "yArea = " << yArea << ", needs to exceed " << aRatio * xProj.size() * yProj.size() << std::endl;

	return abs(correlation) > cRatio
		&& xMean > hRatio * yProj.size()
		&& yMean > hRatio * xProj.size()
		&& xArea > aRatio * xProj.size() * yProj.size()
		&& yArea > aRatio * xProj.size() * yProj.size();
}