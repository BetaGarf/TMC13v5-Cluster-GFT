#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <iostream>
using namespace std;
using namespace cv;
void cvFitPlane(const CvMat* points, float* plane);
double computeDistance(const double A, const double B, const double C, const double D,
	                   const double X, const double Y, const double Z);
