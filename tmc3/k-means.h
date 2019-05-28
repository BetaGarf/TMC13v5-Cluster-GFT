/***************************************************************************
Module Name:
	KMeans
***************************************************************************/

#pragma once
#include <fstream>
#include <PCCPointSet.h>
class KMeans {
public:
  double** means;
  double* min;
  enum InitMode
  {
    InitRandom,
    InitUniform,
  };

  KMeans(int clusterNum = 1);
  ~KMeans();

  void SetMean(int i, const double* u)
  {
    memcpy(m_means[i], u, sizeof(double) * 3);
  }
  void SetInitMode(int i) { m_initMode = i; }
  void SetMaxIterNum(int i) { m_maxIterNum = i;}
  void SetEndError(double f) { m_endError = f; }

  double* GetMean(int i) { return m_means[i]; }
  int GetInitMode() { return m_initMode; }
  int GetMaxIterNum() { return m_maxIterNum; }
  double GetEndError() { return m_endError; }

  void Init(pcc::PCCPointSet3& pointCloud);
  void Cluster(pcc::PCCPointSet3& pointCloud, std::vector<int>& ClusterIndex);
  friend std::ostream& operator<<(std::ostream& out, KMeans& kmeans);

private:
  int m_clusterNum;
  double** m_means;

  int m_initMode;
  int
    m_maxIterNum;  // The stopping criterion regarding the number of iterations
  double m_endError;  // The stopping criterion regarding the error

  double GetIndex(const double* x, int& index);
  void GetMeans(const double* sample, int& index);
  double CalcDistance(const double* x, const double* u, int dimNum);
};
