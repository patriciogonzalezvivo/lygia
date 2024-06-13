#include "make.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition basic vector operations for float2, float3, and float4 types, to match GLSL's behavior.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_OPERATIONS
#define FNC_OPERATIONS

// ////////////////////////////////////////////////////////////////////////////////
// // assign
// ////////////////////////////////////////////////////////////////////////////////

// #ifdef GLM_VERSION
// inline __host__ __device__ void operator=(float2 &a, glm::vec2 b) { a = make_float2(b); }
// inline __host__ __device__ void operator=(float3 &a, glm::vec3 b) { a = make_float3(b); }
// inline __host__ __device__ void operator=(float4 &a, glm::vec4 b) { a = make_float4(b); }
// #endif

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 &a) { return make_float2(-a.x, -a.y); }
inline __host__ __device__ int2 operator-(int2 &a) { return make_int2(-a.x, -a.y); }
inline __host__ __device__ float3 operator-(float3 &a) { return make_float3(-a.x, -a.y, -a.z); }
inline __host__ __device__ int3 operator-(int3 &a) { return make_int3(-a.x, -a.y, -a.z); }
inline __host__ __device__ float4 operator-(float4 &a) { return make_float4(-a.x, -a.y, -a.z, -a.w); }
inline __host__ __device__ int4 operator-(int4 &a) { return make_int4(-a.x, -a.y, -a.z, -a.w); }

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ void operator+=(float2 &a, float2 b) { a.x += b.x; a.y += b.y; }
inline __host__ __device__ float2 operator+(float2 a, float b) { return make_float2(a.x + b, a.y + b); }
inline __host__ __device__ float2 operator+(float b, float2 a) { return make_float2(a.x + b, a.y + b); }
inline __host__ __device__ void operator+=(float2 &a, float b) { a.x += b; a.y += b; }

inline __host__ __device__ int2 operator+(int2 a, int2 b) { return make_int2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ void operator+=(int2 &a, int2 b) { a.x += b.x; a.y += b.y; }
inline __host__ __device__ int2 operator+(int2 a, int b) { return make_int2(a.x + b, a.y + b); }
inline __host__ __device__ int2 operator+(int b, int2 a) { return make_int2(a.x + b, a.y + b); }
inline __host__ __device__ void operator+=(int2 &a, int b) { a.x += b; a.y += b; }

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b) { return make_uint2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ void operator+=(uint2 &a, uint2 b) { a.x += b.x; a.y += b.y; }
inline __host__ __device__ uint2 operator+(uint2 a, uint b) { return make_uint2(a.x + b, a.y + b); }
inline __host__ __device__ uint2 operator+(uint b, uint2 a) { return make_uint2(a.x + b, a.y + b); }
inline __host__ __device__ void operator+=(uint2 &a, uint b) { a.x += b; a.y += b; }

inline __host__ __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ void operator+=(float3 &a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
inline __host__ __device__ float3 operator+(float3 a, float b) { return make_float3(a.x + b, a.y + b, a.z + b); }
inline __host__ __device__ void operator+=(float3 &a, float b) { a.x += b; a.y += b; a.z += b; }

inline __host__ __device__ int3 operator+(int3 a, int3 b) { return make_int3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ void operator+=(int3 &a, int3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
inline __host__ __device__ int3 operator+(int3 a, int b) { return make_int3(a.x + b, a.y + b, a.z + b); }
inline __host__ __device__ void operator+=(int3 &a, int b) { a.x += b; a.y += b; a.z += b; }

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b) { return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ void operator+=(uint3 &a, uint3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
inline __host__ __device__ uint3 operator+(uint3 a, uint b) { return make_uint3(a.x + b, a.y + b, a.z + b); }
inline __host__ __device__ void operator+=(uint3 &a, uint b) { a.x += b; a.y += b; a.z += b; }

inline __host__ __device__ int3 operator+(int b, int3 a) { return make_int3(a.x + b, a.y + b, a.z + b); }
inline __host__ __device__ uint3 operator+(uint b, uint3 a) { return make_uint3(a.x + b, a.y + b, a.z + b); }
inline __host__ __device__ float3 operator+(float b, float3 a) { return make_float3(a.x + b, a.y + b, a.z + b); }

inline __host__ __device__ float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline __host__ __device__ void operator+=(float4 &a, float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
inline __host__ __device__ float4 operator+(float4 a, float b) { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline __host__ __device__ float4 operator+(float b, float4 a) { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline __host__ __device__ void operator+=(float4 &a, float b) { a.x += b; a.y += b; a.z += b; a.w += b; }

inline __host__ __device__ int4 operator+(int4 a, int4 b) { return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline __host__ __device__ void operator+=(int4 &a, int4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
inline __host__ __device__ int4 operator+(int4 a, int b) { return make_int4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline __host__ __device__ int4 operator+(int b, int4 a) { return make_int4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline __host__ __device__ void operator+=(int4 &a, int b) { a.x += b; a.y += b; a.z += b; a.w += b; }

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b) { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline __host__ __device__ void operator+=(uint4 &a, uint4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
inline __host__ __device__ uint4 operator+(uint4 a, uint b) { return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline __host__ __device__ uint4 operator+(uint b, uint4 a) { return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline __host__ __device__ void operator+=(uint4 &a, uint b) { a.x += b; a.y += b; a.z += b; a.w += b; }

#ifdef GLM_VERSION
inline __host__ __device__ float2 operator+(float2 a, glm::vec2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ void operator+=(float2 &a, glm::vec2 b) { a.x += b.x; a.y += b.y; }
// inline __host__ __device__ float2 operator+(float2 a, float b) { return make_float2(a.x + b, a.y + b); }
// inline __host__ __device__ float2 operator+(float b, float2 a) { return make_float2(a.x + b, a.y + b); }
// inline __host__ __device__ void operator+=(float2 &a, float b) { a.x += b; a.y += b; }

// inline __host__ __device__ int2 operator+(int2 a, int2 b) { return make_int2(a.x + b.x, a.y + b.y); }
// inline __host__ __device__ void operator+=(int2 &a, int2 b) { a.x += b.x; a.y += b.y; }
// inline __host__ __device__ int2 operator+(int2 a, int b) { return make_int2(a.x + b, a.y + b); }
// inline __host__ __device__ int2 operator+(int b, int2 a) { return make_int2(a.x + b, a.y + b); }
// inline __host__ __device__ void operator+=(int2 &a, int b) { a.x += b; a.y += b; }

// inline __host__ __device__ uint2 operator+(uint2 a, uint2 b) { return make_uint2(a.x + b.x, a.y + b.y); }
// inline __host__ __device__ void operator+=(uint2 &a, uint2 b) { a.x += b.x; a.y += b.y; }
// inline __host__ __device__ uint2 operator+(uint2 a, uint b) { return make_uint2(a.x + b, a.y + b); }
// inline __host__ __device__ uint2 operator+(uint b, uint2 a) { return make_uint2(a.x + b, a.y + b); }
// inline __host__ __device__ void operator+=(uint2 &a, uint b) { a.x += b; a.y += b; }

// inline __host__ __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
// inline __host__ __device__ void operator+=(float3 &a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
// inline __host__ __device__ float3 operator+(float3 a, float b) { return make_float3(a.x + b, a.y + b, a.z + b); }
// inline __host__ __device__ void operator+=(float3 &a, float b) { a.x += b; a.y += b; a.z += b; }

// inline __host__ __device__ int3 operator+(int3 a, int3 b) { return make_int3(a.x + b.x, a.y + b.y, a.z + b.z); }
// inline __host__ __device__ void operator+=(int3 &a, int3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
// inline __host__ __device__ int3 operator+(int3 a, int b) { return make_int3(a.x + b, a.y + b, a.z + b); }
// inline __host__ __device__ void operator+=(int3 &a, int b) { a.x += b; a.y += b; a.z += b; }

// inline __host__ __device__ uint3 operator+(uint3 a, uint3 b) { return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z); }
// inline __host__ __device__ void operator+=(uint3 &a, uint3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
// inline __host__ __device__ uint3 operator+(uint3 a, uint b) { return make_uint3(a.x + b, a.y + b, a.z + b); }
// inline __host__ __device__ void operator+=(uint3 &a, uint b) { a.x += b; a.y += b; a.z += b; }

// inline __host__ __device__ int3 operator+(int b, int3 a) { return make_int3(a.x + b, a.y + b, a.z + b); }
// inline __host__ __device__ uint3 operator+(uint b, uint3 a) { return make_uint3(a.x + b, a.y + b, a.z + b); }
// inline __host__ __device__ float3 operator+(float b, float3 a) { return make_float3(a.x + b, a.y + b, a.z + b); }

// inline __host__ __device__ float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
// inline __host__ __device__ void operator+=(float4 &a, float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
// inline __host__ __device__ float4 operator+(float4 a, float b) { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
// inline __host__ __device__ float4 operator+(float b, float4 a) { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
// inline __host__ __device__ void operator+=(float4 &a, float b) { a.x += b; a.y += b; a.z += b; a.w += b; }

// inline __host__ __device__ int4 operator+(int4 a, int4 b) { return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
// inline __host__ __device__ void operator+=(int4 &a, int4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
// inline __host__ __device__ int4 operator+(int4 a, int b) { return make_int4(a.x + b, a.y + b, a.z + b, a.w + b); }
// inline __host__ __device__ int4 operator+(int b, int4 a) { return make_int4(a.x + b, a.y + b, a.z + b, a.w + b); }
// inline __host__ __device__ void operator+=(int4 &a, int b) { a.x += b; a.y += b; a.z += b; a.w += b; }

// inline __host__ __device__ uint4 operator+(uint4 a, uint4 b) { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
// inline __host__ __device__ void operator+=(uint4 &a, uint4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
// inline __host__ __device__ uint4 operator+(uint4 a, uint b) { return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b); }
// inline __host__ __device__ uint4 operator+(uint b, uint4 a) { return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b); }
// inline __host__ __device__ void operator+=(uint4 &a, uint b) { a.x += b; a.y += b; a.z += b; a.w += b; }
#endif

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
inline __host__ __device__ void operator-=(float2 &a, float2 b) { a.x -= b.x; a.y -= b.y; }
inline __host__ __device__ float2 operator-(float2 a, float b) { return make_float2(a.x - b, a.y - b); }
inline __host__ __device__ float2 operator-(float b, float2 a) { return make_float2(b - a.x, b - a.y); }
inline __host__ __device__ void operator-=(float2 &a, float b) { a.x -= b; a.y -= b; }

inline __host__ __device__ int2 operator-(int2 a, int2 b) { return make_int2(a.x - b.x, a.y - b.y); }
inline __host__ __device__ void operator-=(int2 &a, int2 b) { a.x -= b.x; a.y -= b.y; }
inline __host__ __device__ int2 operator-(int2 a, int b) { return make_int2(a.x - b, a.y - b); }
inline __host__ __device__ int2 operator-(int b, int2 a) { return make_int2(b - a.x, b - a.y); }
inline __host__ __device__ void operator-=(int2 &a, int b) { a.x -= b; a.y -= b; }

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b) { return make_uint2(a.x - b.x, a.y - b.y); }
inline __host__ __device__ void operator-=(uint2 &a, uint2 b) { a.x -= b.x; a.y -= b.y; }
inline __host__ __device__ uint2 operator-(uint2 a, uint b) { return make_uint2(a.x - b, a.y - b); }
inline __host__ __device__ uint2 operator-(uint b, uint2 a) { return make_uint2(b - a.x, b - a.y); }
inline __host__ __device__ void operator-=(uint2 &a, uint b) { a.x -= b; a.y -= b; }

inline __host__ __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __host__ __device__ void operator-=(float3 &a, float3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; }
inline __host__ __device__ float3 operator-(float3 a, float b) { return make_float3(a.x - b, a.y - b, a.z - b); }
inline __host__ __device__ float3 operator-(float b, float3 a) { return make_float3(b - a.x, b - a.y, b - a.z); }
inline __host__ __device__ void operator-=(float3 &a, float b) { a.x -= b; a.y -= b; a.z -= b; }

inline __host__ __device__ int3 operator-(int3 a, int3 b) { return make_int3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __host__ __device__ void operator-=(int3 &a, int3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; }
inline __host__ __device__ int3 operator-(int3 a, int b) { return make_int3(a.x - b, a.y - b, a.z - b); }
inline __host__ __device__ int3 operator-(int b, int3 a) { return make_int3(b - a.x, b - a.y, b - a.z); }
inline __host__ __device__ void operator-=(int3 &a, int b) { a.x -= b; a.y -= b; a.z -= b; }

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b) { return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __host__ __device__ void operator-=(uint3 &a, uint3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; }
inline __host__ __device__ uint3 operator-(uint3 a, uint b) { return make_uint3(a.x - b, a.y - b, a.z - b); }
inline __host__ __device__ uint3 operator-(uint b, uint3 a) { return make_uint3(b - a.x, b - a.y, b - a.z); }
inline __host__ __device__ void operator-=(uint3 &a, uint b) { a.x -= b; a.y -= b; a.z -= b; }

inline __host__ __device__ float4 operator-(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
inline __host__ __device__ void operator-=(float4 &a, float4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; }
inline __host__ __device__ float4 operator-(float4 a, float b) { return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); }
inline __host__ __device__ float4 operator-(float a, float4 b) { return make_float4(a - b.x, a - b.y, a - b.z, a - b.w); }
inline __host__ __device__ void operator-=(float4 &a, float b) { a.x -= b; a.y -= b; a.z -= b; a.w -= b; }

inline __host__ __device__ int4 operator-(int4 a, int4 b) { return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
inline __host__ __device__ void operator-=(int4 &a, int4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; }
inline __host__ __device__ int4 operator-(int4 a, int b) { return make_int4(a.x - b, a.y - b, a.z - b, a.w - b); }
inline __host__ __device__ int4 operator-(int b, int4 a) { return make_int4(b - a.x, b - a.y, b - a.z, b - a.w); }
inline __host__ __device__ void operator-=(int4 &a, int b) { a.x -= b; a.y -= b; a.z -= b; a.w -= b; }

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b) { return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
inline __host__ __device__ void operator-=(uint4 &a, uint4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; }
inline __host__ __device__ uint4 operator-(uint4 a, uint b) { return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b); }
inline __host__ __device__ uint4 operator-(uint b, uint4 a) { return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w); }
inline __host__ __device__ void operator-=(uint4 &a, uint b) { a.x -= b; a.y -= b; a.z -= b; a.w -= b; }

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __host__ __device__ void operator*=(float2 &a, float2 b) { a.x *= b.x; a.y *= b.y; }
inline __host__ __device__ float2 operator*(float2 a, float b) { return make_float2(a.x * b, a.y * b); }
inline __host__ __device__ float2 operator*(float b, float2 a) { return make_float2(b * a.x, b * a.y); }
inline __host__ __device__ void operator*=(float2 &a, float b) { a.x *= b; a.y *= b; }

inline __host__ __device__ int2 operator*(int2 a, int2 b) { return make_int2(a.x * b.x, a.y * b.y); }
inline __host__ __device__ void operator*=(int2 &a, int2 b) { a.x *= b.x; a.y *= b.y; }
inline __host__ __device__ int2 operator*(int2 a, int b) { return make_int2(a.x * b, a.y * b); }
inline __host__ __device__ int2 operator*(int b, int2 a) { return make_int2(b * a.x, b * a.y); }
inline __host__ __device__ void operator*=(int2 &a, int b) { a.x *= b; a.y *= b; }

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b) { return make_uint2(a.x * b.x, a.y * b.y); }
inline __host__ __device__ void operator*=(uint2 &a, uint2 b) { a.x *= b.x; a.y *= b.y; }
inline __host__ __device__ uint2 operator*(uint2 a, uint b) { return make_uint2(a.x * b, a.y * b); }
inline __host__ __device__ uint2 operator*(uint b, uint2 a) { return make_uint2(b * a.x, b * a.y); }
inline __host__ __device__ void operator*=(uint2 &a, uint b) { a.x *= b; a.y *= b; }

inline __host__ __device__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline __host__ __device__ void operator*=(float3 &a, float3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; }
inline __host__ __device__ float3 operator*(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
inline __host__ __device__ float3 operator*(float b, float3 a) { return make_float3(b * a.x, b * a.y, b * a.z); }
inline __host__ __device__ void operator*=(float3 &a, float b) { a.x *= b; a.y *= b; a.z *= b; }

inline __host__ __device__ int3 operator*(int3 a, int3 b) { return make_int3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline __host__ __device__ void operator*=(int3 &a, int3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; }
inline __host__ __device__ int3 operator*(int3 a, int b) { return make_int3(a.x * b, a.y * b, a.z * b); }
inline __host__ __device__ int3 operator*(int b, int3 a) { return make_int3(b * a.x, b * a.y, b * a.z); }
inline __host__ __device__ void operator*=(int3 &a, int b) { a.x *= b; a.y *= b; a.z *= b; }

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b) { return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline __host__ __device__ void operator*=(uint3 &a, uint3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; }
inline __host__ __device__ uint3 operator*(uint3 a, uint b) { return make_uint3(a.x * b, a.y * b, a.z * b); }
inline __host__ __device__ uint3 operator*(uint b, uint3 a) { return make_uint3(b * a.x, b * a.y, b * a.z); }
inline __host__ __device__ void operator*=(uint3 &a, uint b) { a.x *= b; a.y *= b; a.z *= b; }

inline __host__ __device__ float4 operator*(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
inline __host__ __device__ void operator*=(float4 &a, float4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; }
inline __host__ __device__ float4 operator*(float4 a, float b) { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
inline __host__ __device__ float4 operator*(float b, float4 a) { return make_float4(b * a.x, b * a.y, b * a.z, b * a.w); }
inline __host__ __device__ void operator*=(float4 &a, float b) { a.x *= b; a.y *= b; a.z *= b; a.w *= b; }

inline __host__ __device__ int4 operator*(int4 a, int4 b) { return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
inline __host__ __device__ void operator*=(int4 &a, int4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; }
inline __host__ __device__ int4 operator*(int4 a, int b) { return make_int4(a.x * b, a.y * b, a.z * b, a.w * b); }
inline __host__ __device__ int4 operator*(int b, int4 a) { return make_int4(b * a.x, b * a.y, b * a.z, b * a.w); }
inline __host__ __device__ void operator*=(int4 &a, int b) { a.x *= b; a.y *= b; a.z *= b; a.w *= b; }

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b) { return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
inline __host__ __device__ void operator*=(uint4 &a, uint4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; }
inline __host__ __device__ uint4 operator*(uint4 a, uint b) { return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b); }
inline __host__ __device__ uint4 operator*(uint b, uint4 a) { return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w); }
inline __host__ __device__ void operator*=(uint4 &a, uint b) { a.x *= b; a.y *= b; a.z *= b; a.w *= b; }

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b) { return make_float2(a.x / b.x, a.y / b.y); }
inline __host__ __device__ void operator/=(float2 &a, float2 b) { a.x /= b.x; a.y /= b.y; }
inline __host__ __device__ float2 operator/(float2 a, float b) { return make_float2(a.x / b, a.y / b); }
inline __host__ __device__ void operator/=(float2 &a, float b) { a.x /= b; a.y /= b; }
inline __host__ __device__ float2 operator/(float b, float2 a) { return make_float2(b / a.x, b / a.y); }

inline __host__ __device__ float3 operator/(float3 a, float3 b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
inline __host__ __device__ void operator/=(float3 &a, float3 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; }
inline __host__ __device__ float3 operator/(float3 a, float b) { return make_float3(a.x / b, a.y / b, a.z / b); }
inline __host__ __device__ void operator/=(float3 &a, float b) { a.x /= b; a.y /= b; a.z /= b; }
inline __host__ __device__ float3 operator/(float b, float3 a) { return make_float3(b / a.x, b / a.y, b / a.z); }

inline __host__ __device__ float4 operator/(float4 a, float4 b) { return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
inline __host__ __device__ void operator/=(float4 &a, float4 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; }
inline __host__ __device__ float4 operator/(float4 a, float b) { return make_float4(a.x / b, a.y / b, a.z / b, a.w / b); }
inline __host__ __device__ void operator/=(float4 &a, float b) { a.x /= b; a.y /= b; a.z /= b; a.w /= b; }
inline __host__ __device__ float4 operator/(float b, float4 a){ return make_float4(b / a.x, b / a.y, b / a.z, b / a.w); }

#endif
