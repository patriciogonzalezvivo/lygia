#include "make.cuh"
#include "dot.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: Unpack a 3D vector into a float. Default base is 256.0
use: <float> unpack(<float3> value [, <float> base])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef UNPACK_FNC
#define UNPACK_FNC unpack256
#endif 

#ifndef FNC_UNPACK
#define FNC_UNPACK

inline __host__ __device__ float unpack8(const float3& value) {
    float3 factor = make_float3( 8.0f, 8.0f * 8.0f, 8.0f * 8.0f * 8.0f );
    return dot(value, factor) / 512.0f;
}

inline __host__ __device__ float unpack16(const float3& value) {
    float3 factor = make_float3( 16.0f, 16.0f * 16.0f, 16.0f * 16.0f * 16.0f );
    return dot(value, factor) / 4096.0f;
}

inline __host__ __device__ float unpack32(const float3& value) {
    float3 factor = make_float3( 32.0f, 32.0f * 32.0f, 32.0f * 32.0f * 32.0f );
    return dot(value, factor) / 32768.0f;
}

inline __host__ __device__ float unpack64(const float3& value) {
    float3 factor = make_float3( 64.0f, 64.0f * 64.0f, 64.0f * 64.0f * 64.0f );
    return dot(value, factor) / 262144.0f;
}

inline __host__ __device__ float unpack128(const float3& value) {
    float3 factor = make_float3( 128.0f, 128.0f * 128.0f, 128.0f * 128.0f * 128.0f );
    return dot(value, factor) / 2097152.0f;
}

inline __host__ __device__ float unpack256(const float3& value) {
    float3 factor = make_float3( 256.0f, 256.0f * 256.0f, 256.0f * 256.0f * 256.0f );
    return dot(value, factor) / 16581375.0f;
}

inline __host__ __device__ float unpack(const float3& value, float base) {
    float base3 = base * base * base;
    float3 factor = make_float3( base, base * base, base3);
    return dot(value, factor) / base3;
}

inline __host__ __device__ float unpack(const float3& value) {
    return UNPACK_FNC(value);
}
#endif