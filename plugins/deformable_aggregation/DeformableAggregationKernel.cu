#include "DeformableAggregationKernel.h"
#include "common.h"
#include <cuda_fp16.h>

template<typename T>
__device__ __forceinline__ T Const(float x) { return T(x); }

template<>
__device__ __forceinline__ __half Const<__half>(float x) {
    return __float2half(x);
}

template <typename T>
__device__ __forceinline__ float ToFloat(T x);

template <>
__device__ __forceinline__ float ToFloat<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float ToFloat<__half>(__half x) {
    return __half2float(x);
}

template<typename FeatureType>
__device__ FeatureType BilinearSampling(
        const FeatureType* __restrict__ bottomData,
        int height,
        int width,
        int numEmbeds,
        FeatureType locH,
        FeatureType locW,
        int basePtr) {

    const float hIm = ToFloat(locH) * static_cast<float>(height) - 0.5F;
    const float wIm = ToFloat(locW) * static_cast<float>(width) - 0.5F;

    const int hLow = floorf(hIm);
    const int wLow = floorf(wIm);
    const int hHigh = hLow + 1;
    const int wHigh = wLow + 1;

    const float lh = hIm - hLow;
    const float lw = wIm - wLow;
    const float hh = 1 - lh, hw = 1 - lw;

    const int wStride = numEmbeds;
    const int hStride = width * wStride;
    const int hLowPtrOffset = hLow * hStride;
    const int hHighPtrOffset = hLowPtrOffset + hStride;
    const int wLowPtrOffset = wLow * wStride;
    const int wHighPtrOffset = wLowPtrOffset + wStride;

    float v1 = 0;
    if (hLow >= 0 && wLow >= 0) {
        const int ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
        v1 = ToFloat(bottomData[ptr1]);
    }
    float v2 = 0;
    if (hLow >= 0 && wHigh <= width - 1) {
        const int ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
        v2 = ToFloat(bottomData[ptr2]);
    }
    float v3 = 0;
    if (hHigh <= height - 1 && wLow >= 0) {
        const int ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
        v3 = ToFloat(bottomData[ptr3]);
    }
    float v4 = 0;
    if (hHigh <= height - 1 && wHigh <= width - 1) {
        const int ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
        v4 = ToFloat(bottomData[ptr4]);
    }

    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return Const<FeatureType>(val);
}

template<typename T>
__device__ __forceinline__ bool LocationNotValid(T loc);

template<>
__device__ __forceinline__ bool LocationNotValid<float>(float loc)
{
    return loc <= 0 || loc >= 1;
}

#define ZERO __ushort_as_half(0x0000)
#define ONE __ushort_as_half(0x3C00)
template<>
__device__ __forceinline__ bool LocationNotValid<__half>(__half loc) {
    return __hle(loc, ZERO) || __hge(loc, ONE);
}

template<typename FeatureType, typename IndexType>
__global__ void deformableAggregationKernel(
        const int numKernels,
        FeatureType* __restrict__ output,
        const FeatureType* __restrict__ mcMsFeat,
        const IndexType* __restrict__ spatialShape,
        const IndexType* __restrict__ scaleStartIndex,
        const FeatureType* __restrict__ sampleLocation,
        const FeatureType* __restrict__ weights,
        int batchSize,
        int numCams,
        int numFeat,
        int numEmbeds,
        int numScale,
        int numAnchors,
        int numPts,
        int numGroups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKernels)
        return;

    const int weightIdx = idx / (numEmbeds / numGroups);
    const auto weight = weights[weightIdx];

    const int channelIndex = idx % numEmbeds;
    idx /= numEmbeds;
    const int scaleIndex = idx % numScale;
    idx /= numScale;

    const int camIndex = idx % numCams;
    idx /= numCams;
    const int ptsIndex = idx % numPts;
    idx /= numPts;

    int anchorIndex = idx % numAnchors;
    idx /= numAnchors;
    const int batchIndex = idx % batchSize;
    idx /= batchSize;

    anchorIndex = batchIndex * numAnchors + anchorIndex;
    const int locOffset = ((anchorIndex * numPts + ptsIndex) * numCams + camIndex) << 1;

    const auto locW = sampleLocation[locOffset];
    if (LocationNotValid(locW))
        return;
    const auto locH = sampleLocation[locOffset + 1];
    if (LocationNotValid(locH))
        return;

    int camScaleIndex = camIndex * numScale + scaleIndex;
    const int value_offset =
            (batchIndex * numFeat + scaleStartIndex[camScaleIndex]) * numEmbeds
            + channelIndex;

    camScaleIndex = camScaleIndex << 1;
    const int h = spatialShape[camScaleIndex];
    const int w = spatialShape[camScaleIndex + 1];

    atomicAdd(output + anchorIndex * numEmbeds + channelIndex,
            BilinearSampling(mcMsFeat, h, w, numEmbeds, locH, locW, value_offset) * weight);
}

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
                               cudaStream_t stream)
{
    int numKernels = batchSize * numCams * numEmbeds * numScale * numAnchors * numPts;
    deformableAggregationKernel<<<(int)ceil(((double)numKernels/128)), 128, 0, stream>>>(
        numKernels,
        output,
        mcMsFeat,
        spatialShape,
        scaleStart,
        sampleLocation,
        weights,
        batchSize,
        numCams,
        numFeat,
        numEmbeds,
        numScale,
        numAnchors,
        numPts,
        numGroups);
        PLUGIN_CUASSERT(cudaGetLastError());
        return 0;
}

// template instantiation
template int32_t deformableAggregationImpl<float, int64_t>(float* __restrict__,float const* __restrict__,int64_t const* __restrict__,int64_t const* __restrict__,float const* __restrict__,float const* __restrict__,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,cudaStream_t);
template int32_t deformableAggregationImpl<__half, int64_t>(__half* __restrict__,__half const* __restrict__,int64_t const* __restrict__,int64_t const* __restrict__,__half const* __restrict__,__half const* __restrict__,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,cudaStream_t);