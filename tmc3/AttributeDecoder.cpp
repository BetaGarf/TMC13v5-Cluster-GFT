/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2017-2018, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the ISO/IEC nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "AttributeDecoder.h"

#include "DualLutCoder.h"
#include "constants.h"
#include "entropy.h"
#include "io_hls.h"
#include "RAHT.h"
#include "FixedPoint.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core.hpp>
#include "ComputePlane.h"

using namespace cv;

#if defined Cluster_LoD
/*-------------------------------修改：头文件-----------------------------------*/
#  include <iostream>
#  include "k-means.h"
#  include <assert.h>
/*-----------------------------------------------------------------------------*/
#endif


namespace pcc {

//============================================================================
// An encapsulation of the entropy decoding methods used in attribute coding
#if defined Cluster_LoD
/*-------------------------------修改：聚类函数-----------------------------------*/
void
DecodeCluster(
  pcc::PCCPointSet3& pointCloud,
  std::vector<int>& ClusterIndex,
  int ClusterNum)
{
  const int pointCount = pointCloud.getPointCount();
  KMeans* kmeans = new KMeans(ClusterNum);
  ClusterIndex.resize(pointCount);
  kmeans->SetInitMode(KMeans::InitUniform);
  std::cout << "开始聚类" << std::endl;
  kmeans->Cluster(pointCloud, ClusterIndex);
  std::cout << "聚类结束" << std::endl;
  delete kmeans;
}
/*-----------------------------------------------------------------------------*/
#endif
struct PCCResidualsDecoder {
  EntropyDecoder arithmeticDecoder;
  StaticBitModel binaryModel0;
  AdaptiveBitModel binaryModelDiff[7];
  AdaptiveBitModel binaryModelIsZero[7];
  AdaptiveBitModel ctxPredMode[2];
  DualLutCoder<false> symbolCoder[2];

  void start(const char* buf, int buf_len);
  void stop();
  int decodePredMode(int max);
  uint32_t decodeSymbol(int k1, int k2);
  void decode(uint32_t values[3]);
  uint32_t decode();
};

//----------------------------------------------------------------------------

void
PCCResidualsDecoder::start(const char* buf, int buf_len)
{
  arithmeticDecoder.setBuffer(buf_len, buf);
  arithmeticDecoder.start();
}

//----------------------------------------------------------------------------

void
PCCResidualsDecoder::stop()
{
  arithmeticDecoder.stop();
}

//----------------------------------------------------------------------------

int
PCCResidualsDecoder::decodePredMode(int maxMode)
{
  int mode = 0;

  if (maxMode == 0)
    return mode;

  int ctxIdx = 0;
  while (arithmeticDecoder.decode(ctxPredMode[ctxIdx])) {
    ctxIdx = 1;
    mode++;
    if (mode == maxMode)
      break;
  }

  return mode;
}

//----------------------------------------------------------------------------

uint32_t
PCCResidualsDecoder::decodeSymbol(int k1, int k2)
{
  if (arithmeticDecoder.decode(binaryModelIsZero[k1])) {
    return 0u;
  }

  uint32_t value = symbolCoder[k2].decode(&arithmeticDecoder);
  if (value == kAttributeResidualAlphabetSize) {
    value +=
      arithmeticDecoder.decodeExpGolomb(0, binaryModel0, binaryModelDiff[k1]);
  }
  ++value;

  return value;
}

//----------------------------------------------------------------------------

void
PCCResidualsDecoder::decode(uint32_t value[3])
{
  value[0] = decodeSymbol(0, 0);
  int b0 = value[0] == 0;
  value[1] = decodeSymbol(1 + b0, 1);
  int b1 = value[1] == 0;
  value[2] = decodeSymbol(3 + (b0 << 1) + b1, 1);
}

//----------------------------------------------------------------------------

uint32_t
PCCResidualsDecoder::decode()
{
  return decodeSymbol(0, 0);
}

//============================================================================
// AttributeDecoder Members

void
AttributeDecoder::decode(
  const AttributeDescription& attr_desc,
  const AttributeParameterSet& attr_aps,
  const PayloadBuffer& payload,
  PCCPointSet3& pointCloud)
{
  int abhSize;
  /* AttributeBrickHeader abh = */ parseAbh(payload, &abhSize);

  PCCResidualsDecoder decoder;
  decoder.start(payload.data() + abhSize, payload.size() - abhSize);

  if (attr_desc.attr_num_dimensions == 1) {
    switch (attr_aps.attr_encoding) {
    case AttributeEncoding::kRAHTransform:
      decodeReflectancesRaht(attr_desc, attr_aps, decoder, pointCloud);
      break;

    case AttributeEncoding::kPredictingTransform:
      decodeReflectancesPred(attr_desc, attr_aps, decoder, pointCloud);
      break;

    case AttributeEncoding::kLiftingTransform:
      decodeReflectancesLift(attr_desc, attr_aps, decoder, pointCloud);
      break;
    }
  } else if (attr_desc.attr_num_dimensions == 3) {
    switch (attr_aps.attr_encoding) {
    case AttributeEncoding::kRAHTransform:
      decodeColorsRaht(attr_desc, attr_aps, decoder, pointCloud);
      break;

    case AttributeEncoding::kPredictingTransform:
      decodeColorsPred(attr_desc, attr_aps, decoder, pointCloud);
      break;

    case AttributeEncoding::kLiftingTransform:
      decodeColorsLift(attr_desc, attr_aps, decoder, pointCloud);
      break;
    }
  } else {
    assert(
      attr_desc.attr_num_dimensions == 1
      || attr_desc.attr_num_dimensions == 3);
  }

  decoder.stop();
}

//----------------------------------------------------------------------------

void
AttributeDecoder::computeReflectancePredictionWeights(
  const AttributeParameterSet& aps,
  const PCCPointSet3& pointCloud,
  const std::vector<uint32_t>& indexes,
  PCCPredictor& predictor,
  PCCResidualsDecoder& decoder)
{
  predictor.computeWeights();
  if (predictor.neighborCount > 1) {
    int64_t minValue = 0;
    int64_t maxValue = 0;
    for (int i = 0; i < predictor.neighborCount; ++i) {
      const uint16_t reflectanceNeighbor = pointCloud.getReflectance(
        indexes[predictor.neighbors[i].predictorIndex]);
      if (i == 0 || reflectanceNeighbor < minValue) {
        minValue = reflectanceNeighbor;
      }
      if (i == 0 || reflectanceNeighbor > maxValue) {
        maxValue = reflectanceNeighbor;
      }
    }
    const int64_t maxDiff = maxValue - minValue;
    if (maxDiff > aps.adaptive_prediction_threshold) {
      predictor.predMode =
        decoder.decodePredMode(aps.max_num_direct_predictors);
    }
  }
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeReflectancesPred(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud)
{
  const size_t pointCount = pointCloud.getPointCount();
  std::vector<PCCPredictor> predictors;
  std::vector<uint32_t> numberOfPointsPerLOD;
  std::vector<uint32_t> indexesLOD;

  if (!aps.lod_binary_tree_enabled_flag) {
    if (aps.num_detail_levels <= 1) {
      buildPredictorsFastNoLod(
        pointCloud, aps.num_pred_nearest_neighbours, aps.search_range,
        predictors, indexesLOD);
    } else {
#if defined Cluster_LoD
      /*----------------------------修改：调用聚类-----------------------------------------*/
      std::cout << "修改后的程序" << std::endl;
      std::vector<int> ClusterIndex;
      int ClusterNum = 5;  //Cluster Number
      DecodeCluster(pointCloud, ClusterIndex, ClusterNum);
      std::cout << "开始LOD划分" << std::endl;
      /*PCCPointSet3 OutputpointCloud;
    OutputpointCloud.resize(pointCount);
    OutputpointCloud.addColors();
    std::vector<pcc::PCCColor3B> color(18);
    uint8_t a[] = {0,   255, 255, 255, 255, 0,   0,   8,   160,
                   128, 47,  72,  124, 189, 222, 250, 219, 176};
    uint8_t b[] = {0,  255, 255, 97,  0,   0,   255, 46,  32,
                   42, 79,  61,  252, 183, 184, 128, 112, 48};
    uint8_t c[] = {0,  0,  255, 0, 0,   255, 0,   84,  240,
                   42, 79, 139, 0, 107, 135, 114, 147, 96};
    for (int i = 0; i < 18; i++) {
      color[i].setColor(a[i], b[i], c[i]);
    }
    for (int i = 0; i < pointCount; i++) {
      OutputpointCloud.setPosition(i, pointCloud[i]);
      OutputpointCloud.setColor(i, color[ClusterIndex[i]]);
    }
    OutputpointCloud.write("E:/pointCode/TMC13V6/build/tmc3/Output.ply", true);*/
      /*----------------------------------------------------------------------------*/
      buildPredictorsFast(
        pointCloud, aps.dist2, aps.num_detail_levels,
        aps.num_pred_nearest_neighbours, aps.search_range, aps.search_range,
        predictors, numberOfPointsPerLOD, indexesLOD,ClusterIndex, ClusterNum);
#else
      buildPredictorsFast(
        pointCloud, aps.dist2, aps.num_detail_levels,
        aps.num_pred_nearest_neighbours, aps.search_range, aps.search_range,
        predictors, numberOfPointsPerLOD, indexesLOD);
#endif
    }
  } else {
    buildLevelOfDetailBinaryTree(pointCloud, numberOfPointsPerLOD, indexesLOD);
    computePredictors(
      pointCloud, numberOfPointsPerLOD, indexesLOD,
      aps.num_pred_nearest_neighbours, predictors);
  }

  const int64_t maxReflectance = (1ll << desc.attr_bitdepth) - 1;
  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    auto& predictor = predictors[predictorIndex];
    const int64_t qs = aps.quant_step_size_luma;
    computeReflectancePredictionWeights(
      aps, pointCloud, indexesLOD, predictor, decoder);
    const uint32_t pointIndex = indexesLOD[predictorIndex];
    uint16_t& reflectance = pointCloud.getReflectance(pointIndex);
    const uint32_t attValue0 = decoder.decode();
    const int64_t quantPredAttValue =
      predictor.predictReflectance(pointCloud, indexesLOD);
    const int64_t delta = PCCInverseQuantization(UIntToInt(attValue0), qs);
    const int64_t reconstructedQuantAttValue = quantPredAttValue + delta;
    reflectance = uint16_t(
      PCCClip(reconstructedQuantAttValue, int64_t(0), maxReflectance));
  }
}

//----------------------------------------------------------------------------

void
AttributeDecoder::computeColorPredictionWeights(
  const AttributeParameterSet& aps,
  const PCCPointSet3& pointCloud,
  const std::vector<uint32_t>& indexes,
  PCCPredictor& predictor,
  PCCResidualsDecoder& decoder)
{
  predictor.computeWeights();
  if (predictor.neighborCount > 1) {
    int64_t minValue[3] = {0, 0, 0};
    int64_t maxValue[3] = {0, 0, 0};
    for (int i = 0; i < predictor.neighborCount; ++i) {
      const PCCColor3B colorNeighbor =
        pointCloud.getColor(indexes[predictor.neighbors[i].predictorIndex]);
      for (size_t k = 0; k < 3; ++k) {
        if (i == 0 || colorNeighbor[k] < minValue[k]) {
          minValue[k] = colorNeighbor[k];
        }
        if (i == 0 || colorNeighbor[k] > maxValue[k]) {
          maxValue[k] = colorNeighbor[k];
        }
      }
    }
    const int64_t maxDiff = (std::max)(
      maxValue[2] - minValue[2],
      (std::max)(maxValue[0] - minValue[0], maxValue[1] - minValue[1]));
    if (maxDiff > aps.adaptive_prediction_threshold) {
#if TransformMode == 0
      predictor.predMode =
        decoder.decodePredMode(aps.max_num_direct_predictors);
#endif
#if TransformMode == 1
      predictor.predMode = 0;
#endif
    }
  }
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeColorsPred(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud)
{
  const size_t pointCount = pointCloud.getPointCount();
  std::vector<PCCPredictor> predictors;
  std::vector<uint32_t> numberOfPointsPerLOD;
  std::vector<uint32_t> indexesLOD;
  std::vector<std::vector<int64_t>> transformGT;
  transformGT.resize(pointCount);
  std::vector<PCCPoint3D> normalForEachPoint;
  normalForEachPoint.resize(pointCount);
  std::vector<PCCColor3B> predictColors;
  predictColors.resize(pointCount);
  double subGraphNodeNum = 50;
  double minPointsNumInEachLOD = 5;
  std::vector<uint32_t> NumberOfPointsInEachLOD;

  if (!aps.lod_binary_tree_enabled_flag) {
    if (aps.num_detail_levels <= 1) {
      buildPredictorsFastNoLod(
        pointCloud, aps.num_pred_nearest_neighbours, aps.search_range,
        predictors, indexesLOD);
    } else {
#if defined Cluster_LoD
      /*----------------------------修改：调用聚类-----------------------------------------*/
      std::cout << "修改后的程序" << std::endl;
      std::vector<int> ClusterIndex;
      int ClusterNum = 5;  //Cluster Number
      DecodeCluster(pointCloud, ClusterIndex, ClusterNum);
      std::cout << "开始LOD划分" << std::endl;
      PCCPointSet3 OutputpointCloud;
      OutputpointCloud.resize(pointCount);
      OutputpointCloud.addColors();
      std::vector<pcc::PCCColor3B> color(18);
      uint8_t a[] = {0,   255, 255, 255, 255, 0,   0,   8,   160,
                     128, 47,  72,  124, 189, 222, 250, 219, 176};
      uint8_t b[] = {0,  255, 255, 97,  0,   0,   255, 46,  32,
                     42, 79,  61,  252, 183, 184, 128, 112, 48};
      uint8_t c[] = {0,  0,  255, 0, 0,   255, 0,   84,  240,
                     42, 79, 139, 0, 107, 135, 114, 147, 96};
      for (int i = 0; i < 18; i++) {
        color[i].setColor(a[i], b[i], c[i]);
      }
      for (int i = 0; i < pointCount; i++) {
        OutputpointCloud.setPosition(i, pointCloud[i]);
        OutputpointCloud.setColor(i, color[ClusterIndex[i]]);
      }
      OutputpointCloud.write(
        "E:/pointCode/TMC13V6/build/tmc3/Output.ply", true);
      /*----------------------------------------------------------------------------*/
      buildPredictorsFast(
        pointCloud, aps.dist2, aps.num_detail_levels,
        aps.num_pred_nearest_neighbours, aps.search_range, aps.search_range,
        predictors, numberOfPointsPerLOD, indexesLOD, ClusterIndex,
        ClusterNum);
#else
      buildPredictorsFast(
        pointCloud, aps.dist2, aps.num_detail_levels,
        aps.num_pred_nearest_neighbours, aps.search_range, aps.search_range,
        predictors, numberOfPointsPerLOD, indexesLOD);
#endif    
    }
  } else {
    buildLevelOfDetailBinaryTree(pointCloud, numberOfPointsPerLOD, indexesLOD);
    computePredictors(
      pointCloud, numberOfPointsPerLOD, indexesLOD,
      aps.num_pred_nearest_neighbours, predictors);
  }

  uint32_t values[3];
  for (size_t i = 0; i < numberOfPointsPerLOD.size(); i++) {
    int numberOfCurrentLOD = (i == 0)
      ? numberOfPointsPerLOD[i]
      : (numberOfPointsPerLOD[i] - numberOfPointsPerLOD[i - 1]);
    NumberOfPointsInEachLOD.push_back(numberOfCurrentLOD);
  }
  // 进行法向量的计算，每个点和其临近点进行曲面拟合，得其法向量，参与后续的曲面距离计算
  for (size_t i = 0; i < NumberOfPointsInEachLOD.size(); i++) {
    int currentLODPointNum = NumberOfPointsInEachLOD[i];
    if (currentLODPointNum > minPointsNumInEachLOD) {
      int neighborNumForPlane = 4;
      int prevLODPointNum = (i == 0) ? 0 : numberOfPointsPerLOD[i - 1];
      for (size_t j = 0; j < currentLODPointNum; j++) {
        //compute the MLS plane
        CvMat* points_mat;
        if (j < neighborNumForPlane / 2) {
          points_mat = cvCreateMat(neighborNumForPlane + 1, 3, CV_32FC1);
          for (size_t k = 0; k <= neighborNumForPlane; ++k) {
            PCCVector3D position =
              pointCloud[indexesLOD[prevLODPointNum + j + k]];
            points_mat->data.fl[k * 3 + 0] = position[0];
            points_mat->data.fl[k * 3 + 1] = position[1];
            points_mat->data.fl[k * 3 + 2] = position[2];
          }
        } else if (j > (currentLODPointNum - neighborNumForPlane / 2 - 1)) {
          points_mat = cvCreateMat(neighborNumForPlane + 1, 3, CV_32FC1);
          for (size_t k = 0; k <= neighborNumForPlane; ++k) {
            PCCVector3D position =
              pointCloud[indexesLOD[prevLODPointNum + j - k]];
            points_mat->data.fl[k * 3 + 0] = position[0];
            points_mat->data.fl[k * 3 + 1] = position[1];
            points_mat->data.fl[k * 3 + 2] = position[2];
          }
        } else {
          points_mat = cvCreateMat(neighborNumForPlane + 1, 3, CV_32FC1);
          for (size_t k = 0; k <= neighborNumForPlane; ++k) {
            PCCVector3D position =
              pointCloud[indexesLOD[prevLODPointNum + j + k - 2]];
            points_mat->data.fl[k * 3 + 0] = position[0];
            points_mat->data.fl[k * 3 + 1] = position[1];
            points_mat->data.fl[k * 3 + 2] = position[2];
          }
        }

        float plane12[4] = {0};  //定义用来储存平面参数的数组
        cvFitPlane(points_mat, plane12);  //调用方程
        double A = plane12[0];
        double B = plane12[1];
        double C = plane12[2];
        double D = -plane12[3];
        PCCPoint3D tempNormal;
        tempNormal[0] = A;
        tempNormal[1] = B;
        tempNormal[2] = C;

        normalForEachPoint[prevLODPointNum + j] = tempNormal;
      }
    }
  }

  for (size_t i = 0; i < NumberOfPointsInEachLOD.size(); i++) {
    double restPointsNumber = (double)NumberOfPointsInEachLOD[i];
    double partitionNum = std::round(restPointsNumber / subGraphNodeNum);
    partitionNum = (partitionNum == 0) ? 1 : partitionNum;
    double tempRestPointsNumber = restPointsNumber;
    int prevLODPointNum = (i == 0) ? 0 : numberOfPointsPerLOD[i - 1];
    // NumberOfPointsInEachLOD[NumberOfPointsInEachLOD.size() - 2]
    if (restPointsNumber > 50) {
      for (size_t q = 0; q < partitionNum; q++) {
        tempRestPointsNumber = tempRestPointsNumber - 50;
        int tempSubGraphNode = 0;
        if (tempRestPointsNumber < 25) {
          //  当前点全部作为子图
          tempSubGraphNode = tempRestPointsNumber + 50;
        } else {
          // 50点作为子图
          tempSubGraphNode = 50;
        }
        // 在LOD中找点建立子图算矩阵
        size_t partitionSize = tempSubGraphNode;
        std::vector<std::vector<double>> weights;
        std::vector<double> eachLineOfWeights;
        std::vector<std::vector<double>> weightsSum;
        std::vector<double> eachLineOfWeightsSum;
        std::vector<double> diagonalOfWeightsSum;
        std::vector<std::vector<double>> laplacianWeight;
        std::vector<double> curvedSurfaceDistanceForEachGraph;
        double dist = 0.0;
        double sumOfWeightsLine;
        double scale = 1;
        double sigma = 0.0;
        double sumOfDistance = 0.0;

        for (size_t j = 0; j < partitionSize; j++) {
          for (size_t k = 0; k < partitionSize; k++) {
            if (j == k) {
              sumOfDistance += 0.0;
            } else {
              int pointJPos = prevLODPointNum + q * 50 + j;
              int pointKPos = prevLODPointNum + q * 50 + k;
              double dotProduct = normalForEachPoint[pointJPos][0]
                  * normalForEachPoint[pointKPos][0]
                + normalForEachPoint[pointJPos][1]
                  * normalForEachPoint[pointKPos][1]
                + normalForEachPoint[pointJPos][2]
                  * normalForEachPoint[pointKPos][2];
              PCCPoint3D basePos;
              basePos[0] = 0;
              basePos[1] = 0;
              basePos[2] = 0;
              double normOfNormalJ =
                (normalForEachPoint[pointJPos] - basePos).getNorm();
              double normOfNormalK =
                (normalForEachPoint[pointKPos] - basePos).getNorm();
              double radOfNormal = std::acos(
                dotProduct
                / (normOfNormalJ * normOfNormalK));  // 法线夹角的弧度
              double angleOfNormal = radOfNormal * 180
                / PI;  // 法线夹角的角度. c++的三角函数传的是弧度，这一步没必要
              radOfNormal =
                (radOfNormal > PI) ? (2 * PI - radOfNormal) : radOfNormal;

              double spaceDistance = (pointCloud[indexesLOD[pointJPos]]
                                      - pointCloud[indexesLOD[pointKPos]])
                                       .getNorm();  // 两点的空间距离
              double curvedSurfaceDistance =
                (isnan(radOfNormal)) ? spaceDistance :  // 两点的近似曲面距离
                (radOfNormal * spaceDistance)
                  / (2 * std::sin(radOfNormal / 2));
              curvedSurfaceDistance = (isnan(curvedSurfaceDistance))
                ? spaceDistance
                : curvedSurfaceDistance;
              double curvedSurfaceDistanceNorm2 =
                curvedSurfaceDistance * curvedSurfaceDistance;
              sumOfDistance += curvedSurfaceDistance;
              curvedSurfaceDistanceForEachGraph.push_back(
                curvedSurfaceDistance);
            }
          }
        }
        double numberOfGraphWeight =
          (double)(partitionSize * (partitionSize - 1));
        double averageDistance = sumOfDistance / numberOfGraphWeight;
        double sumOfDistanceSquare = 0.0;
        for (size_t m = 0; m < curvedSurfaceDistanceForEachGraph.size(); m++) {
          sumOfDistanceSquare +=
            pow((curvedSurfaceDistanceForEachGraph[m] - averageDistance), 2);
        }
        double varianceOfGraphDistance =
          sumOfDistanceSquare / curvedSurfaceDistanceForEachGraph.size();

        //  基于曲面距离方差，sigma已经平方过了,效果不错
        sigma = 0.2 * varianceOfGraphDistance;
        //  tau的取值和距离的平均相关
        double tau = 3.5 * averageDistance;
        //  权重模型的计算，考虑曲面距离的近似 + 类高斯分布的径向基函数
        for (size_t j = 0; j < partitionSize; j++) {
          sumOfWeightsLine = 0.0;
          for (size_t k = 0; k < partitionSize; k++) {
            if (j == k) {
              eachLineOfWeights.push_back(0.0);
            } else {
              int pointJPos = prevLODPointNum + q * 50 + j;
              int pointKPos = prevLODPointNum + q * 50 + k;
              double dotProduct = normalForEachPoint[pointJPos][0]
                  * normalForEachPoint[pointKPos][0]
                + normalForEachPoint[pointJPos][1]
                  * normalForEachPoint[pointKPos][1]
                + normalForEachPoint[pointJPos][2]
                  * normalForEachPoint[pointKPos][2];
              PCCPoint3D basePos;
              basePos[0] = 0;
              basePos[1] = 0;
              basePos[2] = 0;
              double normOfNormalJ =
                (normalForEachPoint[pointJPos] - basePos).getNorm();
              double normOfNormalK =
                (normalForEachPoint[pointKPos] - basePos).getNorm();
              double radOfNormal = std::acos(
                dotProduct
                / (normOfNormalJ * normOfNormalK));  // 法线夹角的弧度
              double angleOfNormal = radOfNormal * 180
                / PI;  // 法线夹角的角度. c++的三角函数传的是弧度，这一步没必要
              radOfNormal =
                (radOfNormal > PI) ? (2 * PI - radOfNormal) : radOfNormal;

              double spaceDistance = (pointCloud[indexesLOD[pointJPos]]
                                      - pointCloud[indexesLOD[pointKPos]])
                                       .getNorm();  // 两点的空间距离
              double curvedSurfaceDistance =
                (isnan(radOfNormal)) ? spaceDistance :  // 两点的近似曲面距离
                (radOfNormal * spaceDistance)
                  / (2 * std::sin(radOfNormal / 2));
              curvedSurfaceDistance = (isnan(curvedSurfaceDistance))
                ? spaceDistance
                : curvedSurfaceDistance;
              double curvedSurfaceDistanceNorm2 =
                curvedSurfaceDistance * curvedSurfaceDistance;
              double sigmaNorm2 = sigma * sigma;
              double distWeight1 = 1 / curvedSurfaceDistance;
              // double distWeight = (curvedSurfaceDistance > tau) ? 0 : std::exp(-1 * curvedSurfaceDistanceNorm2 / sigma);
              double distWeight = 1 / spaceDistance;
              eachLineOfWeights.push_back(distWeight);
              sumOfWeightsLine += distWeight;
            }
          }
          weights.push_back(eachLineOfWeights);
          diagonalOfWeightsSum.push_back(sumOfWeightsLine);
          eachLineOfWeights.clear();
        }
        reverse(diagonalOfWeightsSum.begin(), diagonalOfWeightsSum.end());

        for (size_t m = 0; m < partitionSize; m++) {
          for (size_t n = 0; n < partitionSize; n++) {
            if (m == n) {
              eachLineOfWeightsSum.push_back(diagonalOfWeightsSum.back());
              diagonalOfWeightsSum.pop_back();
            } else {
              eachLineOfWeightsSum.push_back(-1 * weights[m][n]);
            }
          }
          laplacianWeight.push_back(eachLineOfWeightsSum);
          eachLineOfWeightsSum.clear();
        }

        /****************** 二维vector转Mat *********************/
        cv::Mat laplacian;
        Mat temp(
          laplacianWeight.size(), laplacianWeight.at(0).size(), CV_64FC1);
        for (size_t p = 0; p < temp.rows; p++)
          for (size_t q = 0; q < temp.cols; q++)
            temp.at<double>(p, q) = laplacianWeight.at(p).at(q);
        temp.copyTo(laplacian);
        cv::Mat eValuesMat;
        cv::Mat eVectorsMatOrigin;
        cv::eigen(laplacian, eValuesMat, eVectorsMatOrigin);
        cv::Mat eVectorsMat = eVectorsMatOrigin * scale;
        cv::Mat inverseEVectorsMatOrigin = eVectorsMat.inv();
        cv::Mat inverseEVectorsMat = inverseEVectorsMatOrigin * scale;
        //  残差的还原
        for (size_t m = 0; m < partitionSize; m++) {
          int pointMPos = prevLODPointNum + q * 50 + m;
          auto& predictor = predictors[pointMPos];
          computeColorPredictionWeights(
            aps, pointCloud, indexesLOD, predictor, decoder);
        }
        for (size_t x = 0; x < partitionSize; x++) {
          int pointXPos = prevLODPointNum + q * 50 + x;
          const int64_t qs = aps.quant_step_size_luma;
          const int64_t qs2 = aps.quant_step_size_chroma;
          const uint32_t pointIndex = indexesLOD[pointXPos];
          auto& predictor = predictors[pointXPos];
          decoder.decode(values);
          /*char yStr[200] = "0";
				  sprintf(yStr, "%d %d %d\n",
				  values[0], values[1], values[2]);
				  fwrite(yStr, strlen(yStr), 1, fp);
				  fflush(fp);*/
          int64_t clipMax = (1 << desc.attr_bitdepth) - 1;
          std::vector<int64_t> tempTransformCoefficient;
          for (size_t k = 0; k < 3; ++k) {
            const int64_t delta =
              PCCInverseQuantization(UIntToInt(values[k]), qs);
            tempTransformCoefficient.push_back(delta);
          }
          transformGT[pointXPos] = tempTransformCoefficient;
        }
        std::vector<double> colorResidualR;
        std::vector<double> colorResidualG;
        std::vector<double> colorResidualB;
        const int64_t qs = aps.quant_step_size_luma;
        for (size_t w = 0; w < partitionSize; w++) {
          int pointWPos = prevLODPointNum + q * 50 + w;
          colorResidualR.push_back(transformGT[pointWPos][0]);
          colorResidualG.push_back(transformGT[pointWPos][1]);
          colorResidualB.push_back(transformGT[pointWPos][2]);
        }
        Mat R(colorResidualR);
        R = R.reshape(0, colorResidualR.size());
        Mat G(colorResidualG);
        G = G.reshape(0, colorResidualG.size());
        Mat B(colorResidualB);
        B = B.reshape(0, colorResidualB.size());
        cv::Mat colorR = inverseEVectorsMat * R;
        cv::Mat colorG = inverseEVectorsMat * G;
        cv::Mat colorB = inverseEVectorsMat * B;
        std::vector<double> colorRVector;
        std::vector<double> colorGVector;
        std::vector<double> colorBVector;
        double clipMax = (1 << desc.attr_bitdepth) - 1;

        for (auto n = 0; n < colorR.rows; n++) {
          for (auto j = 0; j < colorR.cols; j++) {
            colorRVector.push_back(colorR.at<double>(n, j));
          }
        }
        for (auto n = 0; n < colorG.rows; n++) {
          for (auto j = 0; j < colorG.cols; j++) {
            colorGVector.push_back(colorG.at<double>(n, j));
          }
        }
        for (auto n = 0; n < colorB.rows; n++) {
          for (auto j = 0; j < colorB.cols; j++) {
            colorBVector.push_back(colorB.at<double>(n, j));
          }
        }
        for (size_t k = 0; k < partitionSize; k++) {
          int pointKPos = prevLODPointNum + q * 50 + k;
          const uint32_t pointIndex = indexesLOD[pointKPos];
          auto& predictor = predictors[pointKPos];
          const PCCColor3B predictedColor =
            predictor.predictColor(pointCloud, indexesLOD);
          PCCColor3B& color = pointCloud.getColor(pointIndex);

          int64_t clipMax = (1 << desc.attr_bitdepth) - 1;
          const int64_t reconstructedQuantAttValueR =
            colorRVector[k] + predictedColor[0];
          const int64_t reconstructedQuantAttValueG =
            colorGVector[k] + predictedColor[1];
          const int64_t reconstructedQuantAttValueB =
            colorBVector[k] + predictedColor[2];
          color[0] =
            uint8_t(PCCClip(reconstructedQuantAttValueR, int64_t(0), clipMax));
          color[1] =
            uint8_t(PCCClip(reconstructedQuantAttValueG, int64_t(0), clipMax));
          color[2] =
            uint8_t(PCCClip(reconstructedQuantAttValueB, int64_t(0), clipMax));

          /*char yStr[200] = "0";
				  sprintf(yStr, "%d %d %d\n",
				  color[0], color[1], color[2]);
				  fwrite(yStr, strlen(yStr), 1, fp);
				  fflush(fp);*/
        }
      }

    } else {
      //  某层LOD点数很少的时候暂不考虑GT，照旧
      uint32_t values[3];
      for (size_t x = 0; x < restPointsNumber; x++) {
        int pointXPos = prevLODPointNum + x;
        auto& predictor = predictors[pointXPos];
        const int64_t qs = aps.quant_step_size_luma;
        const int64_t qs2 = aps.quant_step_size_chroma;
        computeColorPredictionWeights(
          aps, pointCloud, indexesLOD, predictor, decoder);
        const uint32_t pointIndex = indexesLOD[pointXPos];
        decoder.decode(values);
        /*char yStr[200] = "0";
			  sprintf(yStr, "%d %d %d\n",
			  values[0], values[1], values[2]);
			  fwrite(yStr, strlen(yStr), 1, fp);
			  fflush(fp);*/
        PCCColor3B& color = pointCloud.getColor(pointIndex);
        const PCCColor3B predictedColor =
          predictor.predictColor(pointCloud, indexesLOD);
        int64_t clipMax = (1 << desc.attr_bitdepth) - 1;
        for (size_t k = 0; k < 3; ++k) {
          const int64_t quantPredAttValue = predictedColor[k];
          const int64_t delta =
            PCCInverseQuantization(UIntToInt(values[k]), qs);
          const int64_t reconstructedQuantAttValue = quantPredAttValue + delta;
          color[k] =
            uint8_t(PCCClip(reconstructedQuantAttValue, int64_t(0), clipMax));
        }
        /*char yStr[200] = "0";
			  sprintf(yStr, "%d %d %d\n",
			  color[0], color[1], color[2]);
			  fwrite(yStr, strlen(yStr), 1, fp);
			  fflush(fp);*/
      }
    }
  }
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeReflectancesRaht(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud)
{
  const int voxelCount = int(pointCloud.getPointCount());
  uint64_t* weight = new uint64_t[voxelCount];
  int* binaryLayer = new int[voxelCount];
  std::vector<MortonCodeWithIndex> packedVoxel(voxelCount);
  for (int n = 0; n < voxelCount; n++) {
    weight[n] = 1;
    const auto position = pointCloud[n];
    int x = int(position[0]);
    int y = int(position[1]);
    int z = int(position[2]);
    long long mortonCode = 0;
    for (int b = 0; b < aps.raht_depth; b++) {
      mortonCode |= (long long)((x >> b) & 1) << (3 * b + 2);
      mortonCode |= (long long)((y >> b) & 1) << (3 * b + 1);
      mortonCode |= (long long)((z >> b) & 1) << (3 * b);
    }
    packedVoxel[n].mortonCode = mortonCode;
    packedVoxel[n].index = n;
  }
  sort(packedVoxel.begin(), packedVoxel.end());

  // Morton codes
  long long* mortonCode = new long long[voxelCount];
  for (int n = 0; n < voxelCount; n++) {
    mortonCode[n] = packedVoxel[n].mortonCode;
  }

  // Entropy decode
  const int attribCount = 1;
  uint32_t value;
  int* integerizedAttributes = new int[attribCount * voxelCount];
  for (int n = 0; n < voxelCount; ++n) {
    value = decoder.decode();
    integerizedAttributes[n] = UIntToInt(value);
  }

  FixedPoint* attributes = new FixedPoint[attribCount * voxelCount];

  regionAdaptiveHierarchicalInverseTransform(
    FixedPoint(aps.quant_step_size_luma), mortonCode, attributes, weight,
    attribCount, voxelCount, integerizedAttributes);

  const int64_t maxReflectance = (1 << desc.attr_bitdepth) - 1;
  const int64_t minReflectance = 0;
  for (int n = 0; n < voxelCount; n++) {
    int64_t val = attributes[attribCount * n].round();
    const uint16_t reflectance =
      (uint16_t)PCCClip(val, minReflectance, maxReflectance);
    pointCloud.setReflectance(packedVoxel[n].index, reflectance);
  }

  // De-allocate arrays.
  delete[] binaryLayer;
  delete[] mortonCode;
  delete[] attributes;
  delete[] integerizedAttributes;
  delete[] weight;
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeColorsRaht(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud)
{
  const int voxelCount = int(pointCloud.getPointCount());
  uint64_t* weight = new uint64_t[voxelCount];
  int* binaryLayer = new int[voxelCount];
  std::vector<MortonCodeWithIndex> packedVoxel(voxelCount);
  for (int n = 0; n < voxelCount; n++) {
    weight[n] = 1;
    const auto position = pointCloud[n];
    int x = int(position[0]);
    int y = int(position[1]);
    int z = int(position[2]);
    long long mortonCode = 0;
    for (int b = 0; b < aps.raht_depth; b++) {
      mortonCode |= (long long)((x >> b) & 1) << (3 * b + 2);
      mortonCode |= (long long)((y >> b) & 1) << (3 * b + 1);
      mortonCode |= (long long)((z >> b) & 1) << (3 * b);
    }
    packedVoxel[n].mortonCode = mortonCode;
    packedVoxel[n].index = n;
  }
  sort(packedVoxel.begin(), packedVoxel.end());

  // Morton codes
  long long* mortonCode = new long long[voxelCount];
  for (int n = 0; n < voxelCount; n++) {
    mortonCode[n] = packedVoxel[n].mortonCode;
  }

  // Entropy decode
  const int attribCount = 3;
  uint32_t values[3];
  int* integerizedAttributes = new int[attribCount * voxelCount];

  for (int n = 0; n < voxelCount; ++n) {
    decoder.decode(values);
    for (int d = 0; d < attribCount; ++d) {
      integerizedAttributes[voxelCount * d + n] = UIntToInt(values[d]);
    }
  }

  FixedPoint* attributes = new FixedPoint[attribCount * voxelCount];

  regionAdaptiveHierarchicalInverseTransform(
    FixedPoint(aps.quant_step_size_luma), mortonCode, attributes, weight,
    attribCount, voxelCount, integerizedAttributes);

  const int clipMax = (1 << desc.attr_bitdepth) - 1;
  for (int n = 0; n < voxelCount; n++) {
    const int r = attributes[attribCount * n].round();
    const int g = attributes[attribCount * n + 1].round();
    const int b = attributes[attribCount * n + 2].round();
    PCCColor3B color;
    color[0] = uint8_t(PCCClip(r, 0, clipMax));
    color[1] = uint8_t(PCCClip(g, 0, clipMax));
    color[2] = uint8_t(PCCClip(b, 0, clipMax));
    pointCloud.setColor(packedVoxel[n].index, color);
  }

  // De-allocate arrays.
  delete[] binaryLayer;
  delete[] mortonCode;
  delete[] attributes;
  delete[] integerizedAttributes;
  delete[] weight;
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeColorsLift(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud)
{
  const size_t pointCount = pointCloud.getPointCount();
  std::vector<PCCPredictor> predictors;
  std::vector<uint32_t> numberOfPointsPerLOD;
  std::vector<uint32_t> indexesLOD;
  std::vector<std::vector<int64_t>> transformGT;
  transformGT.resize(pointCount);
  std::vector<PCCPoint3D> normalForEachPoint;
  normalForEachPoint.resize(pointCount);
  std::vector<PCCColor3B> predictColors;
  predictColors.resize(pointCount);
  double subGraphNodeNum = 50;
  double minPointsNumInEachLOD = 5;
  
  if (!aps.lod_binary_tree_enabled_flag) {
    buildPredictorsFast(
      pointCloud, aps.dist2, aps.num_detail_levels,
      aps.num_pred_nearest_neighbours, aps.search_range, aps.search_range,
      predictors, numberOfPointsPerLOD, indexesLOD);
  } else {
#if defined Cluster_LoD
    /*----------------------------修改：调用聚类-----------------------------------------*/
    std::cout << "修改后的程序" << std::endl;
    std::vector<int> ClusterIndex;
    int ClusterNum = 5;  //Cluster Number
    DecodeCluster(pointCloud, ClusterIndex, ClusterNum);
    std::cout << "开始LOD划分" << std::endl;
    PCCPointSet3 OutputpointCloud;
    OutputpointCloud.resize(pointCount);
    OutputpointCloud.addColors();
    std::vector<pcc::PCCColor3B> color(18);
    uint8_t a[] = {0,   255, 255, 255, 255, 0,   0,   8,   160,
                   128, 47,  72,  124, 189, 222, 250, 219, 176};
    uint8_t b[] = {0,  255, 255, 97,  0,   0,   255, 46,  32,
                   42, 79,  61,  252, 183, 184, 128, 112, 48};
    uint8_t c[] = {0,  0,  255, 0, 0,   255, 0,   84,  240,
                   42, 79, 139, 0, 107, 135, 114, 147, 96};
    for (int i = 0; i < 18; i++) {
      color[i].setColor(a[i], b[i], c[i]);
    }
    for (int i = 0; i < pointCount; i++) {
      OutputpointCloud.setPosition(i, pointCloud[i]);
      OutputpointCloud.setColor(i, color[ClusterIndex[i]]);
    }
    OutputpointCloud.write("E:/pointCode/TMC13V6/build/tmc3/Output.ply", true);
    /*----------------------------------------------------------------------------*/
    buildPredictorsFast(
      pointCloud, aps.dist2, aps.num_detail_levels,
      aps.num_pred_nearest_neighbours, aps.search_range, aps.search_range,
      predictors, numberOfPointsPerLOD, indexesLOD, ClusterIndex, ClusterNum);
#else
    buildPredictorsFast(
      pointCloud, aps.dist2, aps.num_detail_levels,
      aps.num_pred_nearest_neighbours, aps.search_range, aps.search_range,
      predictors, numberOfPointsPerLOD, indexesLOD);
#endif    
  }

  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    predictors[predictorIndex].computeWeights();
  }
  std::vector<double> weights;
  PCCComputeQuantizationWeights(predictors, weights);
  const size_t lodCount = numberOfPointsPerLOD.size();
  std::vector<PCCVector3D> colors;
  colors.resize(pointCount);
  // decompress
  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    uint32_t values[3];
    decoder.decode(values);
    const int64_t qs = aps.quant_step_size_luma;
    const int64_t qs2 = aps.quant_step_size_chroma;
    const double quantWeight = sqrt(weights[predictorIndex]);
    auto& color = colors[predictorIndex];
    const int64_t delta = UIntToInt(values[0]);
    const double reconstructedDelta = PCCInverseQuantization(delta, qs);
    color[0] = reconstructedDelta / quantWeight;
    for (size_t d = 1; d < 3; ++d) {
      const int64_t delta = UIntToInt(values[d]);
      const double reconstructedDelta = PCCInverseQuantization(delta, qs2);
      color[d] = reconstructedDelta / quantWeight;
    }
  }

  // reconstruct
  for (size_t lodIndex = 1; lodIndex < lodCount; ++lodIndex) {
    const size_t startIndex = numberOfPointsPerLOD[lodIndex - 1];
    const size_t endIndex = numberOfPointsPerLOD[lodIndex];
    PCCLiftUpdate(predictors, weights, startIndex, endIndex, false, colors);
    PCCLiftPredict(predictors, startIndex, endIndex, false, colors);
  }

  const double clipMax = (1 << desc.attr_bitdepth) - 1;
  for (size_t f = 0; f < pointCount; ++f) {
    PCCColor3B color;
    for (size_t d = 0; d < 3; ++d) {
      color[d] = uint8_t(PCCClip(std::round(colors[f][d]), 0.0, clipMax));
    }
    pointCloud.setColor(indexesLOD[f], color);
  }
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeReflectancesLift(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud)
{
  const size_t pointCount = pointCloud.getPointCount();
  std::vector<PCCPredictor> predictors;
  std::vector<uint32_t> numberOfPointsPerLOD;
  std::vector<uint32_t> indexesLOD;

  if (!aps.lod_binary_tree_enabled_flag) {
    buildPredictorsFast(
      pointCloud, aps.dist2, aps.num_detail_levels,
      aps.num_pred_nearest_neighbours, aps.search_range, aps.search_range,
      predictors, numberOfPointsPerLOD, indexesLOD);
  } else {
    buildLevelOfDetailBinaryTree(pointCloud, numberOfPointsPerLOD, indexesLOD);
    computePredictors(
      pointCloud, numberOfPointsPerLOD, indexesLOD,
      aps.num_pred_nearest_neighbours, predictors);
  }

  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    predictors[predictorIndex].computeWeights();
  }
  std::vector<double> weights;
  PCCComputeQuantizationWeights(predictors, weights);
  const size_t lodCount = numberOfPointsPerLOD.size();
  std::vector<double> reflectances;
  reflectances.resize(pointCount);

  // decompress
  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    const int64_t detail = decoder.decode();
    const int64_t qs = aps.quant_step_size_luma;
    const double quantWeight = sqrt(weights[predictorIndex]);
    auto& reflectance = reflectances[predictorIndex];
    const int64_t delta = UIntToInt(detail);
    const double reconstructedDelta = PCCInverseQuantization(delta, qs);
    reflectance = reconstructedDelta / quantWeight;
  }

  // reconstruct
  for (size_t lodIndex = 1; lodIndex < lodCount; ++lodIndex) {
    const size_t startIndex = numberOfPointsPerLOD[lodIndex - 1];
    const size_t endIndex = numberOfPointsPerLOD[lodIndex];
    PCCLiftUpdate(
      predictors, weights, startIndex, endIndex, false, reflectances);
    PCCLiftPredict(predictors, startIndex, endIndex, false, reflectances);
  }
  const double maxReflectance = (1 << desc.attr_bitdepth) - 1;
  for (size_t f = 0; f < pointCount; ++f) {
    pointCloud.setReflectance(
      indexesLOD[f],
      uint16_t(PCCClip(std::round(reflectances[f]), 0.0, maxReflectance)));
  }
}

//============================================================================

} /* namespace pcc */
