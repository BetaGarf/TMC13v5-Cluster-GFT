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

#include "AttributeEncoder.h"

#include "ArithmeticCodec.h"
#include "DualLutCoder.h"
#include "constants.h"
#include "entropy.h"
#include "RAHT.h"
#include "FixedPoint.h"
#include "ComputePlane.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core.hpp>

using namespace cv;

#if defined Cluster_LoD
/*-------------------------------修改：头文件-----------------------------------*/
#  include <iostream>
#  include "k-means.h"
#  include <assert.h>
/*-----------------------------------------------------------------------------*/
#endif

// todo(df): promote to per-attribute encoder parameter
static const float kAttrPredLambdaR = 0.01;
static const float kAttrPredLambdaC = 0.01;

namespace pcc {
//============================================================================
// An encapsulation of the entropy coding methods used in attribute coding

#if defined Cluster_LoD
/*-------------------------------修改：聚类函数-----------------------------------*/
void
EncodeCluster(
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

struct PCCResidualsEncoder {
  EntropyEncoder arithmeticEncoder;
  StaticBitModel binaryModel0;
  AdaptiveBitModel binaryModelDiff[7];
  AdaptiveBitModel binaryModelIsZero[7];
  AdaptiveBitModel ctxPredMode[2];
  DualLutCoder<false> symbolCoder[2];

  void start(int numPoints);
  int stop();
  void encodePredMode(int value, int max);
  void encodeSymbol(uint32_t value, int k1, int k2);
  void encode(uint32_t value0, uint32_t value1, uint32_t value2);
  void encode(uint32_t value);
};

//----------------------------------------------------------------------------

void
PCCResidualsEncoder::start(int pointCount)
{
  // todo(df): remove estimate when arithmetic codec is replaced
  int maxAcBufLen = pointCount * 3 * 2 + 1024;
  arithmeticEncoder.setBuffer(maxAcBufLen, nullptr);
  arithmeticEncoder.start();
}

//----------------------------------------------------------------------------

int
PCCResidualsEncoder::stop()
{
  return arithmeticEncoder.stop();
}

//----------------------------------------------------------------------------

void
PCCResidualsEncoder::encodePredMode(int mode, int maxMode)
{
  // max = 0 => no direct predictors are used
  if (maxMode == 0)
    return;

  int ctxIdx = 0;
  for (int i = 0; i < mode; i++) {
    arithmeticEncoder.encode(1, ctxPredMode[ctxIdx]);
    ctxIdx = 1;
  }

  // Truncated unary
  if (mode != maxMode)
    arithmeticEncoder.encode(0, ctxPredMode[ctxIdx]);
}

//----------------------------------------------------------------------------

void
PCCResidualsEncoder::encodeSymbol(uint32_t value, int k1, int k2)
{
  bool isZero = value == 0;
  arithmeticEncoder.encode(isZero, binaryModelIsZero[k1]);
  if (isZero) {
    return;
  }
  --value;
  if (value < kAttributeResidualAlphabetSize) {
    symbolCoder[k2].encode(value, &arithmeticEncoder);
  } else {
    int alphabetSize = kAttributeResidualAlphabetSize;
    symbolCoder[k2].encode(alphabetSize, &arithmeticEncoder);
    arithmeticEncoder.encodeExpGolomb(
      value - alphabetSize, 0, binaryModel0, binaryModelDiff[k1]);
  }
}

//----------------------------------------------------------------------------

void
PCCResidualsEncoder::encode(uint32_t value0, uint32_t value1, uint32_t value2)
{
  int b0 = value0 == 0;
  int b1 = value1 == 0;
  encodeSymbol(value0, 0, 0);
  encodeSymbol(value1, 1 + b0, 1);
  encodeSymbol(value2, 3 + (b0 << 1) + b1, 1);
}

//----------------------------------------------------------------------------

void
PCCResidualsEncoder::encode(uint32_t value)
{
  encodeSymbol(value, 0, 0);
}

//============================================================================
// An encapsulation of the entropy coding methods used in attribute coding

struct PCCResidualsEntropyEstimator {
  size_t freq0[kAttributeResidualAlphabetSize + 1];
  size_t freq1[kAttributeResidualAlphabetSize + 1];
  size_t symbolCount0;
  size_t symbolCount1;
  size_t isZero0Count;
  size_t isZero1Count;
  PCCResidualsEntropyEstimator() { init(); }
  void init();
  double bitsDetail(
    const uint32_t detail,
    const size_t symbolCount,
    const size_t* const freq) const;
  double bits(const uint32_t value0) const;
  void update(const uint32_t value0);
  double bits(
    const uint32_t value0, const uint32_t value1, const uint32_t value2) const;
  void
  update(const uint32_t value0, const uint32_t value1, const uint32_t value2);
};

//----------------------------------------------------------------------------

void
PCCResidualsEntropyEstimator::init()
{
  for (size_t i = 0; i <= kAttributeResidualAlphabetSize; ++i) {
    freq0[i] = 1;
    freq1[i] = 1;
  }
  symbolCount0 = kAttributeResidualAlphabetSize + 1;
  symbolCount1 = kAttributeResidualAlphabetSize + 1;
  isZero1Count = isZero0Count = symbolCount0 / 2;
}

//----------------------------------------------------------------------------

double
PCCResidualsEntropyEstimator::bitsDetail(
  const uint32_t detail,
  const size_t symbolCount,
  const size_t* const freq) const
{
  const uint32_t detailClipped =
    std::min(detail, uint32_t(kAttributeResidualAlphabetSize));
  const double pDetail =
    PCCClip(double(freq[detailClipped]) / symbolCount, 0.001, 0.999);
  double bits = -log2(pDetail);
  if (detail >= kAttributeResidualAlphabetSize) {
    const double x = double(detail) - double(kAttributeResidualAlphabetSize);
    bits += 2.0 * std::floor(log2(x + 1.0)) + 1.0;
  }
  return bits;
}

//----------------------------------------------------------------------------

double
PCCResidualsEntropyEstimator::bits(const uint32_t value0) const
{
  const bool isZero0 = value0 == 0;
  const double pIsZero0 = isZero0
    ? double(isZero0Count) / symbolCount0
    : double(symbolCount0 - isZero0Count) / symbolCount0;
  double bits = -log2(PCCClip(pIsZero0, 0.001, 0.999));
  if (!isZero0) {
    bits += bitsDetail(value0 - 1, symbolCount0, freq0);
  }
  return bits;
}

//----------------------------------------------------------------------------

void
PCCResidualsEntropyEstimator::update(const uint32_t value0)
{
  const bool isZero0 = value0 == 0;
  ++symbolCount0;
  if (!isZero0) {
    ++freq0[std::min(value0 - 1, uint32_t(kAttributeResidualAlphabetSize))];
  } else {
    ++isZero0Count;
  }
}

//----------------------------------------------------------------------------

double
PCCResidualsEntropyEstimator::bits(
  const uint32_t value0, const uint32_t value1, const uint32_t value2) const
{
  const bool isZero0 = value0 == 0;
  const double pIsZero0 = isZero0
    ? double(isZero0Count) / symbolCount0
    : double(symbolCount0 - isZero0Count) / symbolCount0;
  double bits = -log2(PCCClip(pIsZero0, 0.001, 0.999));
  if (!isZero0) {
    bits += bitsDetail(value0 - 1, symbolCount0, freq0);
  }

  const bool isZero1 = value1 == 0 && value2 == 0;
  const double pIsZero1 = isZero1
    ? double(isZero1Count) / symbolCount0
    : double(symbolCount0 - isZero1Count) / symbolCount0;
  bits -= log2(PCCClip(pIsZero1, 0.001, 0.999));
  if (!isZero1) {
    bits += bitsDetail(value1, symbolCount1, freq1);
    bits += bitsDetail(value2, symbolCount1, freq1);
  }
  return bits;
}

//----------------------------------------------------------------------------

void
PCCResidualsEntropyEstimator::update(
  const uint32_t value0, const uint32_t value1, const uint32_t value2)
{
  const bool isZero0 = value0 == 0;
  ++symbolCount0;
  if (!isZero0) {
    ++freq0[std::min(value0 - 1, uint32_t(kAttributeResidualAlphabetSize))];
  } else {
    ++isZero0Count;
  }

  const bool isZero1 = value1 == 0 && value2 == 0;
  symbolCount1 += 2;
  if (!isZero1) {
    ++freq1[std::min(value1, uint32_t(kAttributeResidualAlphabetSize))];
    ++freq1[std::min(value2, uint32_t(kAttributeResidualAlphabetSize))];
  } else {
    ++isZero1Count;
  }
}

//============================================================================
// AttributeEncoder Members

void
AttributeEncoder::encode(
  const AttributeDescription& desc,
  const AttributeParameterSet& attr_aps,
  PCCPointSet3& pointCloud,
  PayloadBuffer* payload)
{
  PCCResidualsEncoder encoder;
  encoder.start(int(pointCloud.getPointCount()));

  if (desc.attr_num_dimensions == 1) {
    switch (attr_aps.attr_encoding) {
    case AttributeEncoding::kRAHTransform:
      encodeReflectancesTransformRaht(desc, attr_aps, pointCloud, encoder);
      break;

    case AttributeEncoding::kPredictingTransform:
      encodeReflectancesPred(desc, attr_aps, pointCloud, encoder);
      break;

    case AttributeEncoding::kLiftingTransform:
      encodeReflectancesLift(desc, attr_aps, pointCloud, encoder);
      break;
    }
  } else if (desc.attr_num_dimensions == 3) {
    switch (attr_aps.attr_encoding) {
    case AttributeEncoding::kRAHTransform:
      encodeColorsTransformRaht(desc, attr_aps, pointCloud, encoder);
      break;

    case AttributeEncoding::kPredictingTransform:
      encodeColorsPred(desc, attr_aps, pointCloud, encoder);
      break;

    case AttributeEncoding::kLiftingTransform:
      encodeColorsLift(desc, attr_aps, pointCloud, encoder);
      break;
    }
  } else {
    assert(desc.attr_num_dimensions == 1 || desc.attr_num_dimensions == 3);
  }

  uint32_t acDataLen = encoder.stop();
  std::copy_n(
    encoder.arithmeticEncoder.buffer(), acDataLen,
    std::back_inserter(*payload));
}

//----------------------------------------------------------------------------

int64_t
AttributeEncoder::computeReflectanceResidual(
  const uint64_t reflectance,
  const uint64_t predictedReflectance,
  const int64_t qs)
{
  const int64_t quantAttValue = reflectance;
  const int64_t quantPredAttValue = predictedReflectance;
  const int64_t delta = PCCQuantization(quantAttValue - quantPredAttValue, qs);
  return IntToUInt(delta);
}

//----------------------------------------------------------------------------

void
AttributeEncoder::computeReflectancePredictionWeights(
  const AttributeParameterSet& aps,
  const PCCPointSet3& pointCloud,
  const std::vector<uint32_t>& indexesLOD,
  const uint32_t predictorIndex,
  PCCPredictor& predictor,
  PCCResidualsEncoder& encoder,
  PCCResidualsEntropyEstimator& context)
{
  predictor.computeWeights();
  if (predictor.neighborCount > 1) {
    int64_t minValue = 0;
    int64_t maxValue = 0;
    for (size_t i = 0; i < predictor.neighborCount; ++i) {
      const uint16_t reflectanceNeighbor = pointCloud.getReflectance(
        indexesLOD[predictor.neighbors[i].predictorIndex]);
      if (i == 0 || reflectanceNeighbor < minValue) {
        minValue = reflectanceNeighbor;
      }
      if (i == 0 || reflectanceNeighbor > maxValue) {
        maxValue = reflectanceNeighbor;
      }
    }
    const int64_t maxDiff = maxValue - minValue;
    if (maxDiff > aps.adaptive_prediction_threshold) {
      const int qs = aps.quant_step_size_luma;
      uint16_t attrValue =
        pointCloud.getReflectance(indexesLOD[predictorIndex]);

      // base case: weighted average of n neighbours
      predictor.predMode = 0;
      uint16_t attrPred = predictor.predictReflectance(pointCloud, indexesLOD);
      int64_t attrResidualQuant =
        computeReflectanceResidual(attrValue, attrPred, qs);

      double best_score = attrResidualQuant + kAttrPredLambdaR * (double)qs;

      for (int i = 0; i < predictor.neighborCount; i++) {
        if (i == aps.max_num_direct_predictors)
          break;

        attrPred = pointCloud.getReflectance(
          indexesLOD[predictor.neighbors[i].predictorIndex]);
        attrResidualQuant =
          computeReflectanceResidual(attrValue, attrPred, qs);

        double idxBits = i + (i == aps.max_num_direct_predictors - 1 ? 1 : 2);
        double score = attrResidualQuant + idxBits * kAttrPredLambdaR * qs;

        if (score < best_score) {
          best_score = score;
          predictor.predMode = i + 1;
          // NB: setting predictor.neighborCount = 1 will cause issues
          // with reconstruction.
        }
      }

      encoder.encodePredMode(
        predictor.predMode, aps.max_num_direct_predictors);
    }
  }
}

//----------------------------------------------------------------------------

void
AttributeEncoder::encodeReflectancesPred(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCPointSet3& pointCloud,
  PCCResidualsEncoder& encoder)
{
  const uint32_t pointCount = pointCloud.getPointCount();
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
      EncodeCluster(pointCloud, ClusterIndex, ClusterNum);
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

  const int64_t clipMax = (1ll << desc.attr_bitdepth) - 1;
  PCCResidualsEntropyEstimator context;
  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    auto& predictor = predictors[predictorIndex];
    const int64_t qs = aps.quant_step_size_luma;
    computeReflectancePredictionWeights(
      aps, pointCloud, indexesLOD, predictorIndex, predictor, encoder,
      context);
    const uint32_t pointIndex = indexesLOD[predictorIndex];
    const uint16_t reflectance = pointCloud.getReflectance(pointIndex);
    const uint16_t predictedReflectance =
      predictor.predictReflectance(pointCloud, indexesLOD);
    const int64_t quantAttValue = reflectance;
    const int64_t quantPredAttValue = predictedReflectance;
    const int64_t delta =
      PCCQuantization(quantAttValue - quantPredAttValue, qs);
    const uint32_t attValue0 = uint32_t(IntToUInt(long(delta)));
    const int64_t reconstructedDelta = PCCInverseQuantization(delta, qs);
    const int64_t reconstructedQuantAttValue =
      quantPredAttValue + reconstructedDelta;
    const uint16_t reconstructedReflectance =
      uint16_t(PCCClip(reconstructedQuantAttValue, int64_t(0), clipMax));

    encoder.encode(attValue0);
    pointCloud.setReflectance(pointIndex, reconstructedReflectance);
  }
}

//----------------------------------------------------------------------------

PCCVector3<int64_t>
AttributeEncoder::computeColorResiduals(
  const PCCColor3B color,
  const PCCColor3B predictedColor,
  const int64_t qs,
  const int64_t qs2)
{
  PCCVector3<int64_t> residuals;
  const int64_t quantAttValue = color[0];
  const int64_t quantPredAttValue = predictedColor[0];
  const int64_t delta = PCCQuantization(quantAttValue - quantPredAttValue, qs);
  residuals[0] = IntToUInt(delta);
  for (size_t k = 1; k < 3; ++k) {
    const int64_t quantAttValue = color[k];
    const int64_t quantPredAttValue = predictedColor[k];
    const int64_t delta =
      PCCQuantization(quantAttValue - quantPredAttValue, qs2);
    residuals[k] = IntToUInt(delta);
  }
  return residuals;
}

//----------------------------------------------------------------------------

void
AttributeEncoder::computeColorPredictionWeights(
  const AttributeParameterSet& aps,
  const PCCPointSet3& pointCloud,
  const std::vector<uint32_t>& indexesLOD,
  const uint32_t predictorIndex,
  PCCPredictor& predictor,
  PCCResidualsEncoder& encoder,
  PCCResidualsEntropyEstimator& context)
{
  predictor.computeWeights();
  if (predictor.neighborCount > 1) {
    int64_t minValue[3] = {0, 0, 0};
    int64_t maxValue[3] = {0, 0, 0};
    for (int i = 0; i < predictor.neighborCount; ++i) {
      const PCCColor3B colorNeighbor =
        pointCloud.getColor(indexesLOD[predictor.neighbors[i].predictorIndex]);
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
      const int qs = aps.quant_step_size_luma;
      const int qs2 = aps.quant_step_size_chroma;
      PCCColor3B attrValue = pointCloud.getColor(indexesLOD[predictorIndex]);

      // base case: weighted average of n neighbours
      predictor.predMode = 0;
      PCCColor3B attrPred = predictor.predictColor(pointCloud, indexesLOD);
      PCCVector3<int64_t> attrResidualQuant =
        computeColorResiduals(attrValue, attrPred, qs, qs2);

      double best_score = attrResidualQuant[0] + attrResidualQuant[1]
        + attrResidualQuant[2] + kAttrPredLambdaC * (double)qs;

      for (int i = 0; i < predictor.neighborCount; i++) {
        if (i == aps.max_num_direct_predictors)
          break;

        attrPred = pointCloud.getColor(
          indexesLOD[predictor.neighbors[i].predictorIndex]);
        attrResidualQuant =
          computeColorResiduals(attrValue, attrPred, qs, qs2);

        double idxBits = i + (i == aps.max_num_direct_predictors - 1 ? 1 : 2);
        double score = attrResidualQuant[0] + attrResidualQuant[1]
          + attrResidualQuant[2] + idxBits * kAttrPredLambdaC * qs;

        if (score < best_score) {
          best_score = score;
          predictor.predMode = i + 1;
          // NB: setting predictor.neighborCount = 1 will cause issues
          // with reconstruction.
        }
      }

      #if TransformMode == 0
      encoder.encodePredMode(
        predictor.predMode, aps.max_num_direct_predictors);
#endif
#if TransformMode == 1
      predictor.predMode = 0;
#endif
    }
  }
}

//----------------------------------------------------------------------------

void
AttributeEncoder::encodeColorsPred(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCPointSet3& pointCloud,
  PCCResidualsEncoder& encoder)
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
      EncodeCluster(pointCloud, ClusterIndex, ClusterNum);
      std::cout << "开始LOD划分" << std::endl;
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
  std::vector<std::vector<int64_t>> residualOfColor;
  residualOfColor.resize(pointCount);
  std::vector<PCCPoint3D> normalForEachPoint;
  normalForEachPoint.resize(pointCount);
  std::vector<std::vector<int64_t>>
    GTCoefficient;  //  按照LOD的顺序来存储各点的残差经过GT的系数的量化后的参数
  GTCoefficient.resize(pointCount);
  std::vector<uint32_t> NumberOfPointsInEachLOD;
  double subGraphNodeNum = 50;
  double minPointsNumInEachLOD = 5;
  const int64_t clipMax = (1ll << desc.attr_bitdepth) - 1;
  uint32_t values[3];
  PCCResidualsEntropyEstimator context;
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
        //  残差的计算
        std::vector<double> colorResidualR;
        std::vector<double> colorResidualG;
        std::vector<double> colorResidualB;
        for (size_t p = 0; p < partitionSize; p++) {
          int pointPPos = prevLODPointNum + q * 50 + p;
          const auto pointIndex = indexesLOD[pointPPos];
          auto& predictor = predictors[pointPPos];
          const int64_t qs = aps.quant_step_size_luma;
          const int64_t qs2 = aps.quant_step_size_chroma; 
          computeColorPredictionWeights(
            aps, pointCloud, indexesLOD, pointPPos, predictor, encoder,
            context);
          const PCCColor3B color = pointCloud.getColor(pointIndex);
          const PCCColor3B predictedColor =
            predictor.predictColor(pointCloud, indexesLOD);
          std::vector<int64_t> residualColor;
          residualColor.resize(3);
          residualColor[0] = (int64_t)color[0] - (int64_t)predictedColor[0];
          residualColor[1] = (int64_t)color[1] - (int64_t)predictedColor[1];
          residualColor[2] = (int64_t)color[2] - (int64_t)predictedColor[2];
          residualOfColor[pointPPos] = residualColor;
          colorResidualR.push_back(residualColor[0]);
          colorResidualG.push_back(residualColor[1]);
          colorResidualB.push_back(residualColor[2]);
        }
        Mat R(colorResidualR);
        R = R.reshape(0, colorResidualR.size());
        Mat G(colorResidualG);
        G = G.reshape(0, colorResidualG.size());
        Mat B(colorResidualB);
        B = B.reshape(0, colorResidualB.size());
        //  残差经过GT的变换系数
        cv::Mat residualR = eVectorsMat * R;
        cv::Mat residualG = eVectorsMat * G;
        cv::Mat residualB = eVectorsMat * B;
        std::vector<double> residualVectorR;
        std::vector<double> residualVectorG;
        std::vector<double> residualVectorB;
        for (auto n = 0; n < residualR.rows; n++) {
          for (auto j = 0; j < residualR.cols; j++) {
            residualVectorR.push_back(residualR.at<double>(n, j));
          }
        }
        for (auto n = 0; n < residualG.rows; n++) {
          for (auto j = 0; j < residualG.cols; j++) {
            residualVectorG.push_back(residualG.at<double>(n, j));
          }
        }
        for (auto n = 0; n < residualB.rows; n++) {
          for (auto j = 0; j < residualB.cols; j++) {
            residualVectorB.push_back(residualB.at<double>(n, j));
          }
        }
        uint32_t values[3];
        const int64_t qs = aps.quant_step_size_luma;
        for (size_t x = 0; x < partitionSize; x++) {
          int pointXPos = prevLODPointNum + q * 50 + x;
          double quantAttValue0 = residualVectorR[x];
          double quantAttValue1 = residualVectorG[x];
          double quantAttValue2 = residualVectorB[x];
          const int64_t delta0 = PCCQuantization(quantAttValue0, qs);
          const int64_t delta1 = PCCQuantization(quantAttValue1, qs);
          const int64_t delta2 = PCCQuantization(quantAttValue2, qs);
          std::vector<int64_t> tempGTCoefficient;
          tempGTCoefficient.push_back(delta0);
          tempGTCoefficient.push_back(delta1);
          tempGTCoefficient.push_back(delta2);
          GTCoefficient[pointXPos] = tempGTCoefficient;
          values[0] = uint32_t(IntToUInt(long(delta0)));
          values[1] = uint32_t(IntToUInt(long(delta1)));
          values[2] = uint32_t(IntToUInt(long(delta2)));
          /*char yStr[200] = "0";
				  sprintf(yStr, "%d %d %d\n",
				  values[0], values[1], values[2]);
				  fwrite(yStr, strlen(yStr), 1, fp);
				  fflush(fp);*/
          encoder.encode(values[0], values[1], values[2]);
        }
        std::vector<double> decodeColorResidualR;
        std::vector<double> decodeColorResidualG;
        std::vector<double> decodeColorResidualB;
        for (size_t g = 0; g < partitionSize; g++) {
          int pointGPos = prevLODPointNum + q * 50 + g;
          const auto pointIndex = indexesLOD[pointGPos];
          auto& predictor = predictors[pointGPos];
          const PCCColor3B predictedColor =
            predictor.predictColor(pointCloud, indexesLOD);
          for (size_t k = 0; k < 3; ++k) {
            int64_t quantPredAttValue = predictedColor[k];
            int64_t delta = GTCoefficient[pointGPos][k];
            int64_t reconstructedDelta = PCCInverseQuantization(delta, qs);
            if (k == 0) {
              decodeColorResidualR.push_back(reconstructedDelta);
            }
            if (k == 1) {
              decodeColorResidualG.push_back(reconstructedDelta);
            }
            if (k == 2) {
              decodeColorResidualB.push_back(reconstructedDelta);
            }
          }
        }
        cv::Mat inverseEVectorsMatOrigin = eVectorsMat.inv();
        cv::Mat inverseEVectorsMat = inverseEVectorsMatOrigin * scale;

        Mat deocdeR(decodeColorResidualR);
        deocdeR = deocdeR.reshape(0, decodeColorResidualR.size());
        Mat deocdeG(decodeColorResidualG);
        deocdeG = deocdeG.reshape(0, decodeColorResidualG.size());
        Mat deocdeB(decodeColorResidualB);
        deocdeB = deocdeB.reshape(0, decodeColorResidualB.size());
        cv::Mat dColorR = inverseEVectorsMat * deocdeR;
        cv::Mat dColorG = inverseEVectorsMat * deocdeG;
        cv::Mat dColorB = inverseEVectorsMat * deocdeB;
        std::vector<double> dColorRVector;
        std::vector<double> dColorGVector;
        std::vector<double> dColorBVector;

        for (auto n = 0; n < dColorR.rows; n++) {
          for (auto j = 0; j < dColorR.cols; j++) {
            dColorRVector.push_back(dColorR.at<double>(n, j));
          }
        }
        for (auto n = 0; n < dColorG.rows; n++) {
          for (auto j = 0; j < dColorG.cols; j++) {
            dColorGVector.push_back(dColorG.at<double>(n, j));
          }
        }
        for (auto n = 0; n < dColorB.rows; n++) {
          for (auto j = 0; j < dColorB.cols; j++) {
            dColorBVector.push_back(dColorB.at<double>(n, j));
          }
        }

        for (size_t g = 0; g < partitionSize; g++) {
          int pointGPos = prevLODPointNum + q * 50 + g;
          const auto pointIndex = indexesLOD[pointGPos];
          auto& predictor = predictors[pointGPos];
          const PCCColor3B predictedColor =
            predictor.predictColor(pointCloud, indexesLOD);
          const PCCColor3B color = pointCloud.getColor(pointIndex);
          PCCColor3B reconstructedColor;
          int64_t quantPredAttValue = predictedColor[0];
          int64_t reconstructedDelta = dColorRVector[g];
          int64_t reconstructedQuantAttValue =
            quantPredAttValue + reconstructedDelta;
          reconstructedColor[0] =
            uint8_t(PCCClip(reconstructedQuantAttValue, int64_t(0), clipMax));
          quantPredAttValue = predictedColor[1];
          reconstructedDelta = dColorGVector[g];
          reconstructedQuantAttValue = quantPredAttValue + reconstructedDelta;
          reconstructedColor[1] =
            uint8_t(PCCClip(reconstructedQuantAttValue, int64_t(0), clipMax));
          quantPredAttValue = predictedColor[2];
          reconstructedDelta = dColorBVector[g];
          reconstructedQuantAttValue = quantPredAttValue + reconstructedDelta;
          reconstructedColor[2] =
            uint8_t(PCCClip(reconstructedQuantAttValue, int64_t(0), clipMax));
          pointCloud.setColor(pointIndex, reconstructedColor);
          /*char yStr[200] = "0";
				  sprintf(yStr, "%d %d %d\n",
				  reconstructedColor[0], reconstructedColor[1], reconstructedColor[2]);
				  fwrite(yStr, strlen(yStr), 1, fp);
				  fflush(fp);*/
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
      const int64_t qs = aps.quant_step_size_luma;
      for (size_t x = 0; x < restPointsNumber; x++) {
        int pointXPos = prevLODPointNum + x;
        auto& predictor = predictors[pointXPos];
        computeColorPredictionWeights(
          aps, pointCloud, indexesLOD, pointXPos, predictor, encoder, context);
        const auto pointIndex = indexesLOD[pointXPos];
        const PCCColor3B color = pointCloud.getColor(pointIndex);
        const PCCColor3B predictedColor =
          predictor.predictColor(pointCloud, indexesLOD);
        const int64_t quantAttValue = color[0];
        const int64_t quantPredAttValue = predictedColor[0];
        const int64_t delta =
          PCCQuantization(quantAttValue - quantPredAttValue, qs);
        const int64_t reconstructedDelta = PCCInverseQuantization(delta, qs);
        const int64_t reconstructedQuantAttValue =
          quantPredAttValue + reconstructedDelta;
        values[0] = uint32_t(IntToUInt(long(delta)));
        PCCColor3B reconstructedColor;
        reconstructedColor[0] =
          uint8_t(PCCClip(reconstructedQuantAttValue, int64_t(0), clipMax));
        for (size_t k = 1; k < 3; ++k) {
          const int64_t quantAttValue = color[k];
          const int64_t quantPredAttValue = predictedColor[k];
          const int64_t delta =
            PCCQuantization(quantAttValue - quantPredAttValue, qs);
          const int64_t reconstructedDelta = PCCInverseQuantization(delta, qs);
          const int64_t reconstructedQuantAttValue =
            quantPredAttValue + reconstructedDelta;
          values[k] = uint32_t(IntToUInt(long(delta)));
          reconstructedColor[k] =
            uint8_t(PCCClip(reconstructedQuantAttValue, int64_t(0), clipMax));
        }
        pointCloud.setColor(pointIndex, reconstructedColor);
        /*char yStr[200] = "0";
			  sprintf(yStr, "%d %d %d  %d %d %d\n",
			  color[0], color[1], color[2], reconstructedColor[0], reconstructedColor[1], reconstructedColor[2]);
			  fwrite(yStr, strlen(yStr), 1, fp);
			  fflush(fp);*/
        /*char yStr[200] = "0";
			  sprintf(yStr, "%d %d %d\n",
			  values[0], values[1], values[2]);
			  fwrite(yStr, strlen(yStr), 1, fp);
			  fflush(fp);*/
        encoder.encode(values[0], values[1], values[2]);
      }
    }
  }
}

//----------------------------------------------------------------------------

void
AttributeEncoder::encodeReflectancesTransformRaht(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCPointSet3& pointCloud,
  PCCResidualsEncoder& encoder)
{
  const int voxelCount = int(pointCloud.getPointCount());
  std::vector<MortonCodeWithIndex> packedVoxel(voxelCount);
  for (int n = 0; n < voxelCount; n++) {
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

  // Allocate arrays.
  long long* mortonCode = new long long[voxelCount];
  const int attribCount = 1;
  FixedPoint* attributes = new FixedPoint[attribCount * voxelCount];
  int* integerizedAttributes = new int[attribCount * voxelCount];
  uint64_t* weight = new uint64_t[voxelCount];
  int* binaryLayer = new int[voxelCount];

  // Populate input arrays.
  for (int n = 0; n < voxelCount; n++) {
    weight[n] = 1;
    mortonCode[n] = packedVoxel[n].mortonCode;
    const auto reflectance = pointCloud.getReflectance(packedVoxel[n].index);
    attributes[attribCount * n] = reflectance;
  }

  // Transform.
  regionAdaptiveHierarchicalTransform(
    FixedPoint(aps.quant_step_size_luma), mortonCode, attributes, weight,
    binaryLayer, attribCount, voxelCount, integerizedAttributes);

  // Entropy encode.
  uint32_t value;
  for (int n = 0; n < voxelCount; ++n) {
    const int64_t detail = IntToUInt(integerizedAttributes[n]);
    assert(detail < std::numeric_limits<uint32_t>::max());
    value = uint32_t(detail);
    encoder.encode(value);
  }

  // local decode
  std::fill_n(attributes, attribCount * voxelCount, FixedPoint(0));
  for (int n = 0; n < voxelCount; n++) {
    mortonCode[n] = packedVoxel[n].mortonCode;
    weight[n] = 1;
  }

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
AttributeEncoder::encodeColorsTransformRaht(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCPointSet3& pointCloud,
  PCCResidualsEncoder& encoder)
{
  const int voxelCount = int(pointCloud.getPointCount());
  std::vector<MortonCodeWithIndex> packedVoxel(voxelCount);
  for (int n = 0; n < voxelCount; n++) {
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

  // Allocate arrays.
  long long* mortonCode = new long long[voxelCount];
  const int attribCount = 3;
  FixedPoint* attributes = new FixedPoint[attribCount * voxelCount];
  int* integerizedAttributes = new int[attribCount * voxelCount];
  uint64_t* weight = new uint64_t[voxelCount];
  int* binaryLayer = new int[voxelCount];

  // Populate input arrays.
  for (int n = 0; n < voxelCount; n++) {
    weight[n] = 1;
    mortonCode[n] = packedVoxel[n].mortonCode;
    const auto color = pointCloud.getColor(packedVoxel[n].index);
    attributes[attribCount * n] = color[0];
    attributes[attribCount * n + 1] = color[1];
    attributes[attribCount * n + 2] = color[2];
  }

  // Transform.
  regionAdaptiveHierarchicalTransform(
    FixedPoint(aps.quant_step_size_luma), mortonCode, attributes, weight,
    binaryLayer, attribCount, voxelCount, integerizedAttributes);

  // Entropy encode.
  uint32_t values[attribCount];
  for (int n = 0; n < voxelCount; ++n) {
    for (int d = 0; d < attribCount; ++d) {
      const int64_t detail =
        IntToUInt(integerizedAttributes[voxelCount * d + n]);
      assert(detail < std::numeric_limits<uint32_t>::max());
      values[d] = uint32_t(detail);
    }
    encoder.encode(values[0], values[1], values[2]);
  }

  // local decode
  std::fill_n(attributes, attribCount * voxelCount, FixedPoint(0));
  for (int n = 0; n < voxelCount; n++) {
    weight[n] = 1;
    mortonCode[n] = packedVoxel[n].mortonCode;
  }

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
AttributeEncoder::encodeColorsLift(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCPointSet3& pointCloud,
  PCCResidualsEncoder& encoder)
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
#if defined Cluster_LoD
    /*----------------------------修改：调用聚类-----------------------------------------*/
    std::cout << "修改后的程序" << std::endl;
    std::vector<int> ClusterIndex;
    int ClusterNum = 5;  //Cluster Number
    EncodeCluster(pointCloud, ClusterIndex, ClusterNum);
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

  for (size_t index = 0; index < pointCount; ++index) {
    const auto& color = pointCloud.getColor(indexesLOD[index]);
    for (size_t d = 0; d < 3; ++d) {
      colors[index][d] = color[d];
    }
  }

  for (size_t i = 0; (i + 1) < lodCount; ++i) {
    const size_t lodIndex = lodCount - i - 1;
    const size_t startIndex = numberOfPointsPerLOD[lodIndex - 1];
    const size_t endIndex = numberOfPointsPerLOD[lodIndex];
    PCCLiftPredict(predictors, startIndex, endIndex, true, colors);
    PCCLiftUpdate(predictors, weights, startIndex, endIndex, true, colors);
  }

  // compress
  const int64_t qs = aps.quant_step_size_luma;
  const size_t qs2 = aps.quant_step_size_chroma;
  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    const double quantWeight = sqrt(weights[predictorIndex]);
    auto& color = colors[predictorIndex];
    const int64_t delta = PCCQuantization(color[0] * quantWeight, qs);
    const int64_t detail = IntToUInt(delta);
    assert(detail < std::numeric_limits<uint32_t>::max());
    const double reconstructedDelta = PCCInverseQuantization(delta, qs);
    color[0] = reconstructedDelta / quantWeight;
    uint32_t values[3];
    values[0] = uint32_t(detail);
    for (size_t d = 1; d < 3; ++d) {
      const int64_t delta = PCCQuantization(color[d] * quantWeight, qs2);
      const int64_t detail = IntToUInt(delta);
      assert(detail < std::numeric_limits<uint32_t>::max());
      const double reconstructedDelta = PCCInverseQuantization(delta, qs2);
      color[d] = reconstructedDelta / quantWeight;
      values[d] = uint32_t(detail);
    }
    encoder.encode(values[0], values[1], values[2]);
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
AttributeEncoder::encodeReflectancesLift(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  PCCPointSet3& pointCloud,
  PCCResidualsEncoder& encoder)
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

  for (size_t index = 0; index < pointCount; ++index) {
    reflectances[index] = pointCloud.getReflectance(indexesLOD[index]);
  }

  for (size_t i = 0; (i + 1) < lodCount; ++i) {
    const size_t lodIndex = lodCount - i - 1;
    const size_t startIndex = numberOfPointsPerLOD[lodIndex - 1];
    const size_t endIndex = numberOfPointsPerLOD[lodIndex];
    PCCLiftPredict(predictors, startIndex, endIndex, true, reflectances);
    PCCLiftUpdate(
      predictors, weights, startIndex, endIndex, true, reflectances);
  }

  // compress
  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    const int64_t qs = aps.quant_step_size_luma;
    const double quantWeight = sqrt(weights[predictorIndex]);
    auto& reflectance = reflectances[predictorIndex];
    const int64_t delta = PCCQuantization(reflectance * quantWeight, qs);
    const int64_t detail = IntToUInt(delta);
    assert(detail < std::numeric_limits<uint32_t>::max());
    const double reconstructedDelta = PCCInverseQuantization(delta, qs);
    reflectance = reconstructedDelta / quantWeight;
    encoder.encode(detail);
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
