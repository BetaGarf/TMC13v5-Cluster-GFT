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

#ifndef PCCTMC3Common_h
#define PCCTMC3Common_h

#include "PCCKdTree.h"
#include "PCCMath.h"
#include "constants.h"

#include "nanoflann.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace pcc {
/*----------------------�޸ģ�Structure for LOD0--------------------*/
#if defined Cluster_LoD
struct IndexBool {
  uint32_t index;
  bool isIndexes = false;
};
#endif
// Structure for sorting weights.
struct WeightWithIndex {
  float weight;
  int index;

  WeightWithIndex() = default;

  WeightWithIndex(const int index, const float weight)
    : weight(weight), index(index)
  {}

  // NB: this definition ranks larger weights before smaller values.
  bool operator<(const WeightWithIndex& rhs) const
  {
    // NB: index used to maintain stable sort
    if (weight == rhs.weight)
      return index < rhs.index;
    return weight > rhs.weight;
  }
};

//---------------------------------------------------------------------------

struct MortonCodeWithIndex {
  uint64_t mortonCode;
  int32_t index;
  bool operator<(const MortonCodeWithIndex& rhs) const
  {
    // NB: index used to maintain stable sort
    if (mortonCode == rhs.mortonCode)
      return index < rhs.index;
    return mortonCode < rhs.mortonCode;
  }
};

//---------------------------------------------------------------------------

struct PCCNeighborInfo {
  double weight;
  uint32_t predictorIndex;
  bool operator<(const PCCNeighborInfo& rhs) const
  {
    return (weight == rhs.weight) ? predictorIndex < rhs.predictorIndex
                                  : weight < rhs.weight;
  }
};

//---------------------------------------------------------------------------

struct PCCPredictor {
  uint32_t neighborCount;
  PCCNeighborInfo neighbors[kAttributePredictionMaxNeighbourCount];
  int8_t predMode;

  PCCColor3B predictColor(
    const PCCPointSet3& pointCloud, const std::vector<uint32_t>& indexes) const
  {
    PCCVector3D predicted(0.0);
    if (predMode > neighborCount) {
      /* nop */
    } else if (predMode > 0) {
      const PCCColor3B color =
        pointCloud.getColor(indexes[neighbors[predMode - 1].predictorIndex]);
      for (size_t k = 0; k < 3; ++k) {
        predicted[k] += color[k];
      }
    } else {
      for (size_t i = 0; i < neighborCount; ++i) {
        const PCCColor3B color =
          pointCloud.getColor(indexes[neighbors[i].predictorIndex]);
        const double w = neighbors[i].weight;
        for (size_t k = 0; k < 3; ++k) {
          predicted[k] += w * color[k];
        }
      }
    }
    return PCCColor3B(
      uint8_t(std::round(predicted[0])), uint8_t(std::round(predicted[1])),
      uint8_t(std::round(predicted[2])));
  }

  uint16_t predictReflectance(
    const PCCPointSet3& pointCloud, const std::vector<uint32_t>& indexes) const
  {
    double predicted(0.0);
    if (predMode > neighborCount) {
      /* nop */
    } else if (predMode > 0) {
      predicted = pointCloud.getReflectance(
        indexes[neighbors[predMode - 1].predictorIndex]);
    } else {
      for (size_t i = 0; i < neighborCount; ++i) {
        predicted += neighbors[i].weight
          * pointCloud.getReflectance(indexes[neighbors[i].predictorIndex]);
      }
    }
    return uint16_t(std::round(predicted));
  }

  void computeWeights()
  {
    if (neighborCount <= 1) {
      neighbors[0].weight = 1.0;
      return;
    }
    double sum = 0.0;
    for (uint32_t n = 0; n < neighborCount; ++n) {
      sum += neighbors[n].weight;
    }
    assert(sum > 0.0);
    for (uint32_t n = 0; n < neighborCount; ++n) {
      neighbors[n].weight /= sum;
    }
  }

  void init(const uint32_t predictorIndex)
  {
    neighborCount = (predictorIndex != PCC_UNDEFINED_INDEX) ? 1 : 0;
    neighbors[0].predictorIndex = predictorIndex;
    neighbors[0].weight = 1.0;
    predMode = 0;
  }

  void init() { neighborCount = 0; }

  void insertNeighbor(
    const uint32_t reference,
    const double weight,
    const uint32_t maxNeighborCount)
  {
    bool sort = false;
    assert(
      maxNeighborCount > 0
      && maxNeighborCount <= kAttributePredictionMaxNeighbourCount);
    if (neighborCount < maxNeighborCount) {
      PCCNeighborInfo& neighborInfo = neighbors[neighborCount];
      neighborInfo.weight = weight;
      neighborInfo.predictorIndex = reference;
      ++neighborCount;
      sort = true;
    } else {
      PCCNeighborInfo& neighborInfo = neighbors[maxNeighborCount - 1];
      if (weight < neighborInfo.weight) {
        neighborInfo.weight = weight;
        neighborInfo.predictorIndex = reference;
        sort = true;
      }
    }
    for (int32_t k = neighborCount - 1; k > 0 && sort; --k) {
      if (neighbors[k] < neighbors[k - 1])
        std::swap(neighbors[k], neighbors[k - 1]);
      else
        return;
    }
  }
};

//---------------------------------------------------------------------------

inline int64_t
PCCQuantization(const int64_t value, const int64_t qs)
{
  const int64_t shift = (qs / 3);
  if (!qs) {
    return value;
  }
  if (value >= 0) {
    return (value + shift) / qs;
  }
  return -((shift - value) / qs);
}

inline int64_t
PCCInverseQuantization(const int64_t value, const int64_t qs)
{
  return qs == 0 ? value : (value * qs);
}

inline int64_t
PCCQuantization(const double value, const int64_t qs)
{
  return int64_t(
    value >= 0.0 ? std::floor(value / qs + 1.0 / 3.0)
                 : -std::floor(-value / qs + 1.0 / 3.0));
}

//---------------------------------------------------------------------------

template<typename T>
void
PCCLiftPredict(
  const std::vector<PCCPredictor>& predictors,
  const size_t startIndex,
  const size_t endIndex,
  const bool direct,
  std::vector<T>& attributes)
{
  const size_t predictorCount = endIndex - startIndex;
  for (size_t index = 0; index < predictorCount; ++index) {
    const size_t predictorIndex = predictorCount - index - 1 + startIndex;
    const auto& predictor = predictors[predictorIndex];
    T predicted(0.0);
    for (size_t i = 0; i < predictor.neighborCount; ++i) {
      const size_t neighborPredIndex = predictor.neighbors[i].predictorIndex;
      const double weight = predictor.neighbors[i].weight;
      assert(neighborPredIndex < startIndex);
      predicted += weight * attributes[neighborPredIndex];
    }
    if (direct) {
      attributes[predictorIndex] -= predicted;
    } else {
      attributes[predictorIndex] += predicted;
    }
  }
}

//---------------------------------------------------------------------------

template<typename T>
void
PCCLiftUpdate(
  const std::vector<PCCPredictor>& predictors,
  const std::vector<double>& quantizationWeights,
  const size_t startIndex,
  const size_t endIndex,
  const bool direct,
  std::vector<T>& attributes)
{
  std::vector<double> updateWeights;
  updateWeights.resize(startIndex, 0.0);
  std::vector<T> updates;
  updates.resize(startIndex);
  for (size_t index = 0; index < startIndex; ++index) {
    updates[index] = 0.0;
  }
  const size_t predictorCount = endIndex - startIndex;
  for (size_t index = 0; index < predictorCount; ++index) {
    const size_t predictorIndex = predictorCount - index - 1 + startIndex;
    const auto& predictor = predictors[predictorIndex];
    const double currentQuantWeight = quantizationWeights[predictorIndex];
    for (size_t i = 0; i < predictor.neighborCount; ++i) {
      const size_t neighborPredIndex = predictor.neighbors[i].predictorIndex;
      const double weight = predictor.neighbors[i].weight * currentQuantWeight;
      //assert(neighborPredIndex < startIndex);
      updateWeights[neighborPredIndex] += weight;
      updates[neighborPredIndex] += weight * attributes[predictorIndex];
    }
  }
  for (size_t predictorIndex = 0; predictorIndex < startIndex;
       ++predictorIndex) {
    const double sumWeights = updateWeights[predictorIndex];
    if (sumWeights > 0.0) {
      const double alpha = 1.0 / sumWeights;
      if (direct) {
        attributes[predictorIndex] += alpha * updates[predictorIndex];
      } else {
        attributes[predictorIndex] -= alpha * updates[predictorIndex];
      }
    }
  }
}

//---------------------------------------------------------------------------

inline void
PCCComputeQuantizationWeights(
  const std::vector<PCCPredictor>& predictors,
  std::vector<double>& quantizationWeights)
{
  const size_t pointCount = predictors.size();
  quantizationWeights.resize(pointCount);
  for (size_t i = 0; i < pointCount; ++i) {
    quantizationWeights[i] = 1.0;
  }
  for (size_t i = 0; i < pointCount; ++i) {
    const size_t predictorIndex = pointCount - i - 1;
    const auto& predictor = predictors[predictorIndex];
    const double currentQuantWeight = quantizationWeights[predictorIndex];
    for (size_t i = 0; i < predictor.neighborCount; ++i) {
      const size_t neighborPredIndex = predictor.neighbors[i].predictorIndex;
      const double weight = predictor.neighbors[i].weight;
      quantizationWeights[neighborPredIndex] += weight * currentQuantWeight;
    }
  }
}

//---------------------------------------------------------------------------

inline uint32_t
FindNeighborWithinDistance(
  const PCCPointSet3& pointCloud,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const int32_t index,
  const double radius2,
  const int32_t searchRange,
  std::vector<uint32_t>& retained)
{
  const auto& point = pointCloud[packedVoxel[index].index];
  const int32_t retainedSize = retained.size();
  int32_t j = retainedSize - 2;
  int32_t k = 0;
  while (j >= 0 && ++k < searchRange) {
    const int32_t index1 = retained[j];
    const int32_t pointIndex1 = packedVoxel[index1].index;
    const auto& point1 = pointCloud[pointIndex1];
    const auto d2 = (point1 - point).getNorm2();
    if (d2 <= radius2) {
      return index1;
    }
    --j;
  }
  return PCC_UNDEFINED_INDEX;
}

//---------------------------------------------------------------------------

inline void
computeNearestNeighbors2(
  const PCCPointSet3& pointCloud,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& retained,
  const int32_t startIndex,
  const int32_t endIndex,
  const int32_t searchRange,
  const int32_t numberOfNearestNeighborsInPrediction,
  std::vector<uint32_t>& indexes,
  std::vector<PCCPredictor>& predictors,
  std::vector<uint32_t>& pointIndexToPredictorIndex,
  int32_t& predIndex)
{
  const int32_t retainedSize = retained.size();
  for (int32_t i = startIndex, j = 0; i < endIndex; ++i) {
    const int32_t index = indexes[i];
    const uint64_t mortonCode = packedVoxel[index].mortonCode;
    const int32_t pointIndex = packedVoxel[index].index;
    const auto& point = pointCloud[pointIndex];
    indexes[i] = pointIndex;
    while (j < retainedSize
           && mortonCode >= packedVoxel[retained[j]].mortonCode)
      ++j;
    j = std::min(retainedSize - 1, j);
    auto& predictor = predictors[--predIndex];
    pointIndexToPredictorIndex[pointIndex] = predIndex;
    predictor.init();
    for (int32_t k = j, r = 0; k >= 0 && r < searchRange; --k, ++r) {
      const int32_t pointIndex1 = packedVoxel[retained[k]].index;
      const auto& point1 = pointCloud[pointIndex1];
      predictor.insertNeighbor(
        pointIndex1, (point - point1).getNorm2(),
        numberOfNearestNeighborsInPrediction);
    }
    for (int32_t k = j + 1, r = 0; k < retainedSize && r < searchRange;
         ++k, ++r) {
      const int32_t pointIndex1 = packedVoxel[retained[k]].index;
      const auto& point1 = pointCloud[pointIndex1];
      predictor.insertNeighbor(
        pointIndex1, (point - point1).getNorm2(),
        numberOfNearestNeighborsInPrediction);
    }
    assert(predictor.neighborCount <= numberOfNearestNeighborsInPrediction);
  }
}

//---------------------------------------------------------------------------

inline void
computeNearestNeighbors(
  const PCCPointSet3& pointCloud,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& retained,
  const int32_t startIndex,
  const int32_t endIndex,
  const int32_t searchRange,
  const int32_t numberOfNearestNeighborsInPrediction,
  std::vector<uint32_t>& indexes,
  std::vector<PCCPredictor>& predictors,
  std::vector<uint32_t>& pointIndexToPredictorIndex,
  int32_t& predIndex,
  std::vector<PCCBox3D>& bBoxes)
{
  const int32_t retainedSize = retained.size();
  const int32_t bucketSize = 8;
  bBoxes.resize((retainedSize + bucketSize - 1) / bucketSize);
  for (int32_t i = 0, b = 0; i < retainedSize; ++b) {
    auto& bBox = bBoxes[b];
    bBox.min = bBox.max = pointCloud[packedVoxel[retained[i++]].index];
    for (int32_t k = 1; k < bucketSize && i < retainedSize; ++k, ++i) {
      const int32_t pointIndex = packedVoxel[retained[i]].index;
      const auto& point = pointCloud[pointIndex];
      for (int32_t p = 0; p < 3; ++p) {
        bBox.min[p] = std::min(bBox.min[p], point[p]);
        bBox.max[p] = std::max(bBox.max[p], point[p]);
      }
    }
  }

  const int32_t index0 = numberOfNearestNeighborsInPrediction - 1;

  for (int32_t i = startIndex, j = 0; i < endIndex; ++i) {
    const int32_t index = indexes[i];
    const uint64_t mortonCode = packedVoxel[index].mortonCode;
    const int32_t pointIndex = packedVoxel[index].index;
    const auto& point = pointCloud[pointIndex];
    indexes[i] = pointIndex;
    while (j < retainedSize
           && mortonCode >= packedVoxel[retained[j]].mortonCode)
      ++j;
    j = std::min(retainedSize - 1, j);
    auto& predictor = predictors[--predIndex];
    pointIndexToPredictorIndex[pointIndex] = predIndex;

    predictor.init();

    const int32_t j0 = std::max(0, j - searchRange);
    const int32_t j1 = std::min(retainedSize, j + searchRange + 1);

    const int32_t bucketIndex0 = j / bucketSize;
    int32_t k0 = std::max(bucketIndex0 * bucketSize, j0);
    int32_t k1 = std::min((bucketIndex0 + 1) * bucketSize, j1);

    for (int32_t k = k0; k < k1; ++k) {
      const int32_t pointIndex1 = packedVoxel[retained[k]].index;
      const auto& point1 = pointCloud[pointIndex1];
      predictor.insertNeighbor(
        pointIndex1, (point - point1).getNorm2(),
        numberOfNearestNeighborsInPrediction);
    }

    for (int32_t s0 = 1, sr = (1 + searchRange / bucketSize); s0 < sr; ++s0) {
      for (int32_t s1 = 0; s1 < 2; ++s1) {
        const int32_t bucketIndex1 =
          s1 == 0 ? bucketIndex0 + s0 : bucketIndex0 - s0;
        if (bucketIndex1 < 0 || bucketIndex1 >= bBoxes.size()) {
          continue;
        }
        if (
          predictor.neighborCount < numberOfNearestNeighborsInPrediction
          || bBoxes[bucketIndex1].getDist2(point)
            < predictor.neighbors[index0].weight) {
          const int32_t k0 = std::max(bucketIndex1 * bucketSize, j0);
          const int32_t k1 = std::min((bucketIndex1 + 1) * bucketSize, j1);
          for (int32_t k = k0; k < k1; ++k) {
            const int32_t pointIndex1 = packedVoxel[retained[k]].index;
            const auto& point1 = pointCloud[pointIndex1];
            predictor.insertNeighbor(
              pointIndex1, (point - point1).getNorm2(),
              numberOfNearestNeighborsInPrediction);
          }
        }
      }
    }
    assert(predictor.neighborCount <= numberOfNearestNeighborsInPrediction);
  }
}

//---------------------------------------------------------------------------
#if defined Cluster_LoD
/*---------------------------------�޸ģ�����subsample-----------------------------*/
inline void
subsample(
  const PCCPointSet3& SubPointCloud,
  const std::vector<MortonCodeWithIndex>& SubPackedVoxel,
  std::vector<uint32_t>& sub_input,
  const double k_radius2,
  const int32_t searchRange,
  std::vector<uint32_t>& sub_retained,
  std::vector<uint32_t>& retained,
  std::vector<uint32_t>& indexes,
  std::vector<pcc::IndexBool>& IsIndexes)
{
  if (sub_input.size() == 1) {
    indexes.push_back(IsIndexes[0].index);
  } else {
    for (const auto index : sub_input) {
      if (sub_retained.empty()) {
        sub_retained.push_back(index);
        retained.push_back(IsIndexes[index].index);
        continue;
      }
      const auto& point = SubPointCloud[SubPackedVoxel[index].index];
      if (
        (SubPointCloud[SubPackedVoxel[sub_retained.back()].index] - point)
            .getNorm2()
          <= k_radius2
        || FindNeighborWithinDistance(
             SubPointCloud, SubPackedVoxel, index, k_radius2, searchRange,
             sub_retained)
          != PCC_UNDEFINED_INDEX) {
        indexes.push_back(IsIndexes[index].index);
      } else {
        sub_retained.push_back(index);
        retained.push_back(IsIndexes[index].index);
      }
    }
  }
}
#endif
inline void
subsample(
  const PCCPointSet3& pointCloud,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& input,
  const double radius2,
  const int32_t searchRange,
  std::vector<uint32_t>& retained,
  std::vector<uint32_t>& indexes)
{
  if (input.size() == 1) {
    indexes.push_back(input[0]);
  } else {
    for (const auto index : input) {
      if (retained.empty()) {
        retained.push_back(index);
        continue;
      }
      const auto& point = pointCloud[packedVoxel[index].index];
      if (
        (pointCloud[packedVoxel[retained.back()].index] - point).getNorm2()
          <= radius2
        || FindNeighborWithinDistance(
             pointCloud, packedVoxel, index, radius2, searchRange, retained)
          != PCC_UNDEFINED_INDEX) {
        indexes.push_back(index);
      } else {
        retained.push_back(index);
      }
    }
  }
}

//---------------------------------------------------------------------------

inline void
computeMortonCodes(
  const PCCPointSet3& pointCloud,
  std::vector<MortonCodeWithIndex>& packedVoxel)
{
  const int32_t pointCount = int32_t(pointCloud.getPointCount());
  packedVoxel.resize(pointCount);
  for (int n = 0; n < pointCount; n++) {
    const auto& position = pointCloud[n];
    packedVoxel[n].mortonCode = mortonAddr(
      int32_t(position[0]), int32_t(position[1]), int32_t(position[2]));
    packedVoxel[n].index = n;
  }
  sort(packedVoxel.begin(), packedVoxel.end());
}

//---------------------------------------------------------------------------

inline void
updatePredictors(
  const std::vector<uint32_t>& pointIndexToPredictorIndex,
  std::vector<PCCPredictor>& predictors)
{
  for (auto& predictor : predictors) {
    if (predictor.neighborCount < 2) {
      predictor.neighbors[0].weight = 1.0;
    } else if (predictor.neighbors[0].weight == 0.0) {
      predictor.neighborCount = 1;
      predictor.neighbors[0].weight = 1.0;
    } else {
      for (int32_t k = 0; k < predictor.neighborCount; ++k) {
        auto& weight = predictor.neighbors[k].weight;
        weight = 1.0 / weight;
      }
    }
    for (int32_t k = 0; k < predictor.neighborCount; ++k) {
      auto& neighbor = predictor.neighbors[k];
      neighbor.predictorIndex =
        pointIndexToPredictorIndex[neighbor.predictorIndex];
    }
  }
}

//---------------------------------------------------------------------------
// LoD generation using Binary-tree
inline void
buildLevelOfDetailBinaryTree(
  const PCCPointSet3& pointCloud,
  std::vector<uint32_t>& numberOfPointsPerLOD,
  std::vector<uint32_t>& indexes)
{
  const uint32_t pointCount = pointCloud.getPointCount();
  uint8_t const btDepth = std::log2(round(pointCount / 2)) - 1;
  PCCKdTree3 kdtree(pointCloud, btDepth);
  kdtree.build();

  bool skipLayer = true;  //if true, skips alternate layer of BT
  std::vector<bool> skipDepth(btDepth + 1, false);
  if (skipLayer) {
    for (int i = skipDepth.size() - 2; i >= 0; i--) {
      skipDepth[i] = !skipDepth[i + 1];  //if true, that layer is skipped
    }
  }

  indexes.resize(pointCount);
  std::vector<bool> visited(pointCount, false);
  uint32_t lod = 0;
  uint32_t start = 0;
  uint32_t end = 0;
  for (size_t i = 0; i < btDepth + 1; i++) {
    start = end;
    end = start + (1 << i);
    if (!skipDepth[i]) {
      for (int j = start; j < end; j++) {
        auto indx = kdtree.searchClosestAvailablePoint(kdtree.nodes[j].centd);
        if (indx != PCC_UNDEFINED_INDEX && !visited[indx]) {
          indexes[lod++] = indx;
          visited[indx] = true;
        }
      }
      numberOfPointsPerLOD.push_back(lod);
    }
  }
  for (size_t i = 0; i < pointCount; i++) {
    if (!visited[i]) {
      indexes[lod++] = i;
    }
  }
  numberOfPointsPerLOD.push_back(lod);
}

//---------------------------------------------------------------------------
struct PointCloudWrapper {
  PointCloudWrapper(
    const PCCPointSet3& pointCloud, const std::vector<uint32_t>& indexes)
    : _pointCloud(pointCloud), _indexes(indexes)
  {}
  inline size_t kdtree_get_point_count() const { return _pointCount; }
  inline void kdtree_set_point_count(const size_t pointCount)
  {
    assert(pointCount < _indexes.size());
    assert(pointCount < _pointCloud.getPointCount());
    _pointCount = pointCount;
  }
  inline double kdtree_get_pt(const size_t idx, int dim) const
  {
    assert(idx < _pointCount && dim < 3);
    return _pointCloud[_indexes[idx]][dim];
  }
  template<class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const
  {
    return false;
  }

  const PCCPointSet3& _pointCloud;
  const std::vector<uint32_t>& _indexes;
  size_t _pointCount = 0;
};

//---------------------------------------------------------------------------
inline void
computePredictors(
  const PCCPointSet3& pointCloud,
  const std::vector<uint32_t>& numberOfPointsPerLOD,
  const std::vector<uint32_t>& indexes,
  const size_t numberOfNearestNeighborsInPrediction,
  std::vector<PCCPredictor>& predictors)
{
  const uint32_t PCCTMC3MaxPredictionNearestNeighborCount = 3;
  const size_t pointCount = pointCloud.getPointCount();
  const size_t lodCount = numberOfPointsPerLOD.size();
  assert(lodCount);
  predictors.resize(pointCount);

  // delta prediction for LOD0
  uint32_t i0 = numberOfPointsPerLOD[0];
  for (uint32_t i = 0; i < i0; ++i) {
    auto& predictor = predictors[i];
    if (i == 0) {
      predictor.init(PCC_UNDEFINED_INDEX);
    } else {
      predictor.init(PCC_UNDEFINED_INDEX);
    }
  }
  PointCloudWrapper pointCloudWrapper(pointCloud, indexes);
  const nanoflann::SearchParams params(10, 0.0f, true);
  size_t indices[PCCTMC3MaxPredictionNearestNeighborCount];
  double sqrDist[PCCTMC3MaxPredictionNearestNeighborCount];
  nanoflann::KNNResultSet<double> resultSet(
    numberOfNearestNeighborsInPrediction);
  for (uint32_t lodIndex = 1; lodIndex < lodCount; ++lodIndex) {
    pointCloudWrapper.kdtree_set_point_count(i0);
    nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, PointCloudWrapper>,
      PointCloudWrapper, 3>
      kdtree(
        3, pointCloudWrapper, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();
    const uint32_t i1 = numberOfPointsPerLOD[lodIndex];

    for (uint32_t i = i0; i < i1; ++i) {
      const uint32_t pointIndex = indexes[i];
      const auto& point = pointCloud[pointIndex];
      auto& predictor = predictors[i];
      resultSet.init(indices, sqrDist);
      kdtree.findNeighbors(resultSet, &point[0], params);
      const uint32_t resultCount = resultSet.size();
      if (sqrDist[0] == 0.0 || resultCount == 1) {
        const uint32_t predIndex = indices[0];
        predictor.init(predIndex);
      } else {
        predictor.neighborCount = resultCount;
        for (size_t n = 0; n < resultCount; ++n) {
          const uint32_t predIndex = indices[n];
          assert(predIndex < i);
          predictor.neighbors[n].predictorIndex = predIndex;
          const uint32_t pointIndex1 = indexes[predIndex];
          const auto& point1 = pointCloud[pointIndex1];
          predictor.neighbors[n].weight = 1.0 / (point - point1).getNorm2();
        }
      }
    }
    i0 = i1;
  }
}
#if defined Cluster_LoD
/*---------------------------�޸ģ�����buildPredictorsFast����LoD����------------------------------------*/
inline void
buildPredictorsFast(
   const PCCPointSet3& pointCloud,
  const std::vector<int64_t>& dist2,
  const int32_t levelOfDetailCount,
  const int32_t numberOfNearestNeighborsInPrediction,
  const int32_t searchRange1,
  const int32_t searchRange2,
  std::vector<PCCPredictor>& predictors,
  std::vector<uint32_t>& numberOfPointsPerLevelOfDetail,
  std::vector<uint32_t>& indexes,
  std::vector<int>& ClusterIndex,
  int ClusterNum)
{
  const int32_t pointCount = int32_t(pointCloud.getPointCount());
  assert(pointCount);

  std::vector<MortonCodeWithIndex> packedVoxel;
  computeMortonCodes(pointCloud, packedVoxel);

  std::vector<uint32_t> retained, input, pointIndexToPredictorIndex;
  pointIndexToPredictorIndex.resize(pointCount);
  retained.reserve(pointCount);
  input.resize(pointCount);
  for (uint32_t i = 0; i < pointCount; ++i) {
    input[i] = i;
  }

  // prepare output buffers
  predictors.resize(pointCount);
  numberOfPointsPerLevelOfDetail.resize(0);
  indexes.resize(0);
  indexes.reserve(pointCount);
  numberOfPointsPerLevelOfDetail.reserve(21);
  numberOfPointsPerLevelOfDetail.push_back(pointCount);

  std::cout << "�ӵ��ƻ���" << std::endl;
  std::vector<PCCPointSet3> SubPointCloud;
  SubPointCloud.resize(ClusterNum);
  std::vector<int> k(ClusterNum, 0);
  std::vector<std::vector<MortonCodeWithIndex>> SubPackedVoxel(ClusterNum);
  std::vector<std::vector<pcc::IndexBool>> IsIndexes(ClusterNum);
  //@SubPointCloud���ӵ���
  for (int Cluster_i = 0; Cluster_i < ClusterNum; Cluster_i++) {
    for (int index_j = 0; index_j < pointCount; index_j++) {
      if (ClusterIndex[packedVoxel[index_j].index] == Cluster_i) {
        SubPointCloud[Cluster_i].resize(k[Cluster_i] + 1);
        SubPointCloud[Cluster_i].setPosition(
          k[Cluster_i], pointCloud[packedVoxel[index_j].index]);
        SubPointCloud[Cluster_i].addColors();
        SubPointCloud[Cluster_i].setColor(
          k[Cluster_i]++, pointCloud.getColor(packedVoxel[index_j].index));
        struct IndexBool temp;
        temp.index = index_j;
        IsIndexes[Cluster_i].push_back(temp);
      }
    }
  }
  std::cout << "���ƻ��ֽ���" << std::endl;
  std::vector<std::vector<uint32_t>> sub_input(ClusterNum);
  std::vector<std::vector<uint32_t>> sub_retained(ClusterNum);
  std::vector<uint32_t> SubPointCount;
  std::vector<pcc::PCCBox3<uint32_t>> BoundBox(ClusterNum);
  std::vector<std::vector<uint32_t>> sub_radius2(ClusterNum);
  std::vector<uint32_t> rate;
  double temp = double(1) / double(levelOfDetailCount + 1);
  SubPointCount.reserve(ClusterNum);
  //����BoundingBox
  for (int Cluster_i = 0; Cluster_i < ClusterNum; Cluster_i++) {
    BoundBox[Cluster_i].min = uint32_t(-1);
    BoundBox[Cluster_i].max = uint32_t(0);
    for (size_t i = 0; i < SubPointCloud[Cluster_i].getPointCount(); ++i) {
      const PCCVector3D point = SubPointCloud[Cluster_i][i];
      for (int k = 0; k < 3; ++k) {
        const uint32_t coord = uint32_t(point[k]);
        if (BoundBox[Cluster_i].max[k] < coord) {
          BoundBox[Cluster_i].max[k] = coord;
        }
        if (BoundBox[Cluster_i].min[k] > coord) {
          BoundBox[Cluster_i].min[k] = coord;
        }
      }
    }
  }
  for (int Cluster_i = 0; Cluster_i < ClusterNum; Cluster_i++) {
    //����Ӧ����
    uint32_t temp_radius =
      (BoundBox[Cluster_i].max - BoundBox[Cluster_i].min).getNorm2();
    rate.push_back(round(std::pow(temp_radius, temp)));
    for (int j = 0; j < levelOfDetailCount; j++) {
      /*if (temp_radius == 0)
        break;*/
      sub_radius2[Cluster_i].push_back(temp_radius);
      temp_radius /= rate[Cluster_i];
    }
    std::reverse(sub_radius2[Cluster_i].begin(), sub_radius2[Cluster_i].end());
    //�ӵ���Morton
    computeMortonCodes(SubPointCloud[Cluster_i], SubPackedVoxel[Cluster_i]);
    uint32_t temp_pointCount = SubPointCloud[Cluster_i].getPointCount();
    SubPointCount.push_back(temp_pointCount);
    //@sub_input����ʼ��
    for (uint32_t i = 0; i < temp_pointCount; ++i) {
      sub_input[Cluster_i].push_back(i);
    }
  }
  std::vector<PCCBox3<double>> bBoxes;
  int32_t predIndex = int32_t(pointCount);

  for (uint32_t lodIndex = 0; !input.empty() && lodIndex <= levelOfDetailCount;
       ++lodIndex) {
    const int32_t startIndex = indexes.size();
    for (uint32_t Cluster_i = 0; Cluster_i < ClusterNum; Cluster_i++) {
      if (lodIndex == levelOfDetailCount) {
        std::reverse(sub_input[Cluster_i].begin(), sub_input[Cluster_i].end());
        for (const auto index : sub_input[Cluster_i]) {
          indexes.push_back(IsIndexes[Cluster_i][index].index);
        }
      } else {
        double k_radius2 = dist2[lodIndex];
        //double k_radius2 = sub_radius2[Cluster_i][lodIndex];
        subsample(
          SubPointCloud[Cluster_i], SubPackedVoxel[Cluster_i],
          sub_input[Cluster_i], k_radius2, searchRange1,
          sub_retained[Cluster_i], retained, indexes, IsIndexes[Cluster_i]);
      }
      sub_input[Cluster_i].resize(0);
      std::swap(sub_retained[Cluster_i], sub_input[Cluster_i]);
    }
    const int32_t endIndex = indexes.size();
    if (retained.empty()) {
      for (int32_t i = startIndex; i < endIndex; ++i) {
        const int32_t index = indexes[i];
        const int32_t pointIndex = packedVoxel[index].index;
        indexes[i] = pointIndex;
        auto& predictor = predictors[--predIndex];
        assert(predIndex >= 0);
        pointIndexToPredictorIndex[pointIndex] = predIndex;
        predictor.init(PCC_UNDEFINED_INDEX);
      }
    } else {
      sort(indexes.begin() + startIndex, indexes.begin() + endIndex);
      sort(retained.begin(), retained.end());
      computeNearestNeighbors(
        pointCloud, packedVoxel, retained, startIndex, endIndex, searchRange2,
        numberOfNearestNeighborsInPrediction, indexes, predictors,
        pointIndexToPredictorIndex, predIndex, bBoxes);
      numberOfPointsPerLevelOfDetail.push_back(retained.size());
      retained.resize(0);
    }
  }
  std::reverse(indexes.begin(), indexes.end());
  updatePredictors(pointIndexToPredictorIndex, predictors);
  std::reverse(
    numberOfPointsPerLevelOfDetail.begin(),
    numberOfPointsPerLevelOfDetail.end());
  //����LoDÿ��ĵ���
  for (size_t lodIndex = numberOfPointsPerLevelOfDetail.size() - 2;
       lodIndex > 0; --lodIndex) {
    if (numberOfPointsPerLevelOfDetail[lodIndex] > pointCount / 2) {
      numberOfPointsPerLevelOfDetail[lodIndex] = numberOfPointsPerLevelOfDetail
        [numberOfPointsPerLevelOfDetail.size() - 1];
      numberOfPointsPerLevelOfDetail.pop_back();
    } else {
      break;
    }
  }
}
#endif
//---------------------------------------------------------------------------
inline void
buildPredictorsFast(
  const PCCPointSet3& pointCloud,
  const std::vector<int64_t>& dist2,
  const int32_t levelOfDetailCount,
  const int32_t numberOfNearestNeighborsInPrediction,
  const int32_t searchRange1,
  const int32_t searchRange2,
  std::vector<PCCPredictor>& predictors,
  std::vector<uint32_t>& numberOfPointsPerLevelOfDetail,
  std::vector<uint32_t>& indexes)
{
  const int32_t pointCount = int32_t(pointCloud.getPointCount());
  assert(pointCount);

  std::vector<MortonCodeWithIndex> packedVoxel;
  computeMortonCodes(pointCloud, packedVoxel);

  std::vector<uint32_t> retained, input, pointIndexToPredictorIndex;
  pointIndexToPredictorIndex.resize(pointCount);
  retained.reserve(pointCount);
  input.resize(pointCount);
  for (uint32_t i = 0; i < pointCount; ++i) {
    input[i] = i;
  }

  // prepare output buffers
  predictors.resize(pointCount);
  numberOfPointsPerLevelOfDetail.resize(0);
  indexes.resize(0);
  indexes.reserve(pointCount);
  numberOfPointsPerLevelOfDetail.reserve(21);
  numberOfPointsPerLevelOfDetail.push_back(pointCount);

  std::vector<PCCBox3D> bBoxes;
  int32_t predIndex = int32_t(pointCount);
  for (uint32_t lodIndex = 0; !input.empty() && lodIndex <= levelOfDetailCount;
       ++lodIndex) {
    const int32_t startIndex = indexes.size();
    if (lodIndex == levelOfDetailCount) {
      std::reverse(input.begin(), input.end());
      for (const auto index : input) {
        indexes.push_back(index);
      }
    } else {
      const double radius2 = dist2[lodIndex];
      subsample(
        pointCloud, packedVoxel, input, radius2, searchRange1, retained,
        indexes);
    }
    const int32_t endIndex = indexes.size();

    if (retained.empty()) {
      for (int32_t i = startIndex; i < endIndex; ++i) {
        const int32_t index = indexes[i];
        const int32_t pointIndex = packedVoxel[index].index;
        indexes[i] = pointIndex;
        auto& predictor = predictors[--predIndex];
        assert(predIndex >= 0);
        pointIndexToPredictorIndex[pointIndex] = predIndex;
        predictor.init(PCC_UNDEFINED_INDEX);
      }
      break;
    } else {
      computeNearestNeighbors(
        pointCloud, packedVoxel, retained, startIndex, endIndex, searchRange2,
        numberOfNearestNeighborsInPrediction, indexes, predictors,
        pointIndexToPredictorIndex, predIndex, bBoxes);
    }
    numberOfPointsPerLevelOfDetail.push_back(retained.size());
    input.resize(0);
    std::swap(retained, input);
  }
  std::reverse(indexes.begin(), indexes.end());
  updatePredictors(pointIndexToPredictorIndex, predictors);
  std::reverse(
    numberOfPointsPerLevelOfDetail.begin(),
    numberOfPointsPerLevelOfDetail.end());
}

//---------------------------------------------------------------------------

inline void
buildPredictorsFastNoLod(
  const PCCPointSet3& pointCloud,
  const int32_t numberOfNearestNeighborsInPrediction,
  const int32_t searchRange,
  std::vector<PCCPredictor>& predictors,
  std::vector<uint32_t>& indexes)
{
  const int32_t pointCount = int32_t(pointCloud.getPointCount());
  assert(pointCount);

  indexes.resize(pointCount);
  // re-order points
  {
    std::vector<MortonCodeWithIndex> packedVoxel;
    computeMortonCodes(pointCloud, packedVoxel);
    for (int32_t i = 0; i < pointCount; ++i) {
      indexes[i] = packedVoxel[i].index;
    }
  }
  predictors.resize(pointCount);
  for (int32_t i = 0; i < pointCount; ++i) {
    const int32_t index = indexes[i];
    const auto& point = pointCloud[index];
    auto& predictor = predictors[i];
    predictor.init();
    const int32_t k0 = std::max(0, i - searchRange);
    const int32_t k1 = i - 1;
    for (int32_t k = k1; k >= k0; --k) {
      const int32_t index1 = indexes[k];
      const auto& point1 = pointCloud[index1];
      predictor.insertNeighbor(
        k, (point - point1).getNorm2(), numberOfNearestNeighborsInPrediction);
    }
    assert(predictor.neighborCount <= numberOfNearestNeighborsInPrediction);
    if (predictor.neighborCount < 2) {
      predictor.neighbors[0].weight = 1.0;
    } else if (predictor.neighbors[0].weight == 0.0) {
      predictor.neighborCount = 1;
      predictor.neighbors[0].weight = 1.0;
    } else {
      for (int32_t k = 0; k < predictor.neighborCount; ++k) {
        auto& weight = predictor.neighbors[k].weight;
        weight = 1.0 / weight;
      }
    }
  }
}

//---------------------------------------------------------------------------

}  // namespace pcc

#endif /* PCCTMC3Common_h */
