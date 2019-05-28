/***************************************************************************
Module Name:
	KMeans
***************************************************************************/
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <assert.h>
#include "k-means.h"
#include <PCCPointSet.h>
#include <vector>
#include <PCCPointset.h>
using namespace std;
#define MAX 10000;

KMeans::KMeans(int clusterNum)
{
  m_clusterNum = clusterNum;
  min = new double[m_clusterNum];
  means = new double*[m_clusterNum];
  m_means = new double*[m_clusterNum];
  for (int i = 0; i < m_clusterNum; i++) {
    means[i] = new double[3];
    memset(means[i], 0, sizeof(double) * 3);
    m_means[i] = new double[3];
    memset(m_means[i], 0, sizeof(double) * 3);
  }

  m_initMode = InitRandom;
  m_maxIterNum = 100;
  m_endError = 0.001;
}

KMeans::~KMeans()
{
  delete[] min;
  for (int i = 0; i < m_clusterNum; i++) {
    delete[] m_means[i];
  }
  delete[] m_means;
}

//N 为特征向量数
void
KMeans::Cluster(pcc::PCCPointSet3& pointCloud, std::vector<int>& ClusterIndex)
{
  int pointCount=pointCloud.getPointCount();
  assert(pointCount >= m_clusterNum);
  // Initialize model
  Init(pointCloud);

  // Recursion
  double* x = new double[3];  // Sample data
  int index = -1;             // Class index
  double iterNum = 0;
  double lastCost = 0;
  double currCost = 0;
  int unchanged = 0;
  bool loop = true;
  int* counts = new int[m_clusterNum];
  double** next_means =
    new double*[m_clusterNum];  // New model for reestimation
  for (int i = 0; i < m_clusterNum; i++) {
    next_means[i] = new double[3];
    min[i] = MAX;
  }

  while (loop) {
    //clean buffer for classification
    memset(counts, 0, sizeof(int) * m_clusterNum);
    for (int i = 0; i < m_clusterNum; i++) {
      memset(next_means[i], 0, sizeof(double) * 3);
    }

    lastCost = currCost;
    currCost = 0;

    // Classification
    for (int i = 0; i < pointCount; i++) {
      for (int j = 0; j < 3; j++)
        x[j] = pointCloud[i][j];

      currCost += GetIndex(x, index);

      counts[index]++;
      for (int d = 0; d < 3; d++) {
        next_means[index][d] += x[d];
      }
    }
    currCost /= pointCount;

    // Reestimation
    for (int i = 0; i < m_clusterNum; i++) {
      if (counts[i] > 0) {
        for (int d = 0; d < 3; d++) {
          next_means[i][d] /= counts[i];
        }
        memcpy(m_means[i], next_means[i], sizeof(double) * 3);
      }
    }

    // Terminal conditions
    iterNum++;
    if (fabs(lastCost - currCost) < m_endError * lastCost) {
      unchanged++;
    }
    if (iterNum >= m_maxIterNum || unchanged >= 3) {
      loop = false;
      cout << "loop=false" << endl;
    }

    //DEBUG
    //cout << "Iter: " << iterNum << ", Average Cost: " << currCost << endl;
  }

  // Output the label file
  for (int i = 0; i < pointCount; i++) {
    for (int j = 0; j < 3; j++)
      x[j] = pointCloud[i][j];
    GetMeans(x, index);
    ClusterIndex[i] = index;
  }
  delete[] counts;
  delete[] x;
  for (int i = 0; i < m_clusterNum; i++) {
    delete[] next_means[i];
  }
  delete[] next_means;
}

void
KMeans::Init(pcc::PCCPointSet3& pointCloud)
{
  int pointCount = pointCloud.getPointCount();
  if (m_initMode == InitRandom) {
    int inteval = pointCount / m_clusterNum;
    double* sample = new double[3];

    // Seed the random-number generator with current time
    srand((unsigned)time(NULL));

    for (int i = 0; i < m_clusterNum; i++) {
      int select = inteval * i + (inteval - 1) * rand() / RAND_MAX;
      for (int j = 0; j < 3; j++) {
		sample[j] = pointCloud[i][j];
      }        
      memcpy(m_means[i], sample, sizeof(double) * 3);
    }

    delete[] sample;
  } else if (m_initMode == InitUniform) {
    double* sample = new double[3];

    for (int i = 0; i < m_clusterNum; i++) {
      int select = i * (pointCount / m_clusterNum);
      for (int j = 0; j < 3; j++) {
        sample[j] = pointCloud[i][j];
      }
      memcpy(m_means[i], sample, sizeof(double) * 3);
    }
    delete[] sample;
    }
}

double
KMeans::GetIndex(const double* sample, int& Index)
{
  double dist = -1;
  for (int i = 0; i < m_clusterNum; i++) {
    double temp = CalcDistance(sample, m_means[i], 3);
    if (temp < dist || dist == -1) {
      dist = temp;
      Index = i;
    }
  }
  return dist;
}

void
KMeans::GetMeans(const double* sample, int& Index)
{
  double dist = -1;
  for (int i = 0; i < m_clusterNum; i++) {
    double temp = CalcDistance(sample, m_means[i], 3);
    if (temp < dist || dist == -1) {
      dist = temp;
      Index = i;
      if (min[i] > dist) {
        min[i] = dist;
        memcpy(means[i], sample, sizeof(double) * 3);
      }
    }
  }
}

double
KMeans::CalcDistance(const double* x, const double* u, int dimNum)
{
  double temp = 0;
  for (int d = 0; d < dimNum; d++) {
    temp += (x[d] - u[d]) * (x[d] - u[d]);
  }
  return sqrt(temp);
}

ostream&
operator<<(ostream& out, KMeans& kmeans)
{
  out << "<KMeans>" << endl;
  out << "<DimNum> " << 3 << " </DimNum>" << endl;
  out << "<ClusterNum> " << kmeans.m_clusterNum << " </CluterNum>" << endl;

  out << "<Mean>" << endl;
  for (int i = 0; i < kmeans.m_clusterNum; i++) {
    for (int d = 0; d < 3; d++) {
      out << kmeans.m_means[i][d] << " ";
    }
    out << endl;
  }
  out << "</Mean>" << endl;

  out << "</KMeans>" << endl;
  return out;
}
