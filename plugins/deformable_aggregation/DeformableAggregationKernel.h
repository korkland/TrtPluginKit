#pragma once

#include <cuda_runtime.h>

template <typename FeatureType, typename IndexType>
int32_t deformableAggregationImpl(FeatureType* __restrict__ output,
                               FeatureType const* __restrict__ mcMsFeat,
                               IndexType const* __restrict__ spatialShape,
                               IndexType const* __restrict__ scaleStart,
                               FeatureType const* __restrict__ sampleLocation,
                               FeatureType const* __restrict__ weights,
                               int32_t batchSize,
                               int32_t numCams,
                               int32_t numFeat,
                               int32_t numEmbeds,
                               int32_t numScale,
                               int32_t numAnchors,
                               int32_t numPts,
                               int32_t numGroups,
                               cudaStream_t stream);