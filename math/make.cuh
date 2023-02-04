#ifndef FNC_MAKE
#define FNC_MAKE

#include <cuda_runtime.h>

typedef unsigned int    uint;
typedef unsigned short  ushort;

typedef float2          vec2;
typedef float3          vec3;
typedef float4          vec4;

typedef int2            ivec2;
typedef int3            ivec3;
typedef int4            ivec4;

typedef uint2           uvec2;
typedef uint3           uvec3;
typedef uint4           uvec4;

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s) { return make_float2(s, s); }
inline __host__ __device__ float2 make_float2(float3 a) { return make_float2(a.x, a.y); }
inline __host__ __device__ float2 make_float2(int2 a) { return make_float2(float(a.x), float(a.y)); }
inline __host__ __device__ float2 make_float2(uint2 a) { return make_float2(float(a.x), float(a.y)); }

inline __host__ __device__ int2 make_int2(int s) { return make_int2(s, s); }
inline __host__ __device__ int2 make_int2(int3 a) { return make_int2(a.x, a.y); }
inline __host__ __device__ int2 make_int2(uint2 a) { return make_int2(int(a.x), int(a.y)); }
inline __host__ __device__ int2 make_int2(float2 a) { return make_int2(int(a.x), int(a.y)); }

inline __host__ __device__ uint2 make_uint2(uint s) { return make_uint2(s, s); }
inline __host__ __device__ uint2 make_uint2(uint3 a) { return make_uint2(a.x, a.y); }
inline __host__ __device__ uint2 make_uint2(int2 a) { return make_uint2(uint(a.x), uint(a.y)); }

inline __host__ __device__ float3 make_float3(float s) { return make_float3(s, s, s); }
inline __host__ __device__ float3 make_float3(float2 a) { return make_float3(a.x, a.y, 0.0f); }
inline __host__ __device__ float3 make_float3(float2 a, float s) { return make_float3(a.x, a.y, s); }
inline __host__ __device__ float3 make_float3(float4 a) { return make_float3(a.x, a.y, a.z); }
inline __host__ __device__ float3 make_float3(int3 a) { return make_float3(float(a.x), float(a.y), float(a.z)); }
inline __host__ __device__ float3 make_float3(uint3 a) { return make_float3(float(a.x), float(a.y), float(a.z)); }

inline __host__ __device__ int3 make_int3(int s) { return make_int3(s, s, s); }
inline __host__ __device__ int3 make_int3(int2 a) { return make_int3(a.x, a.y, 0); }
inline __host__ __device__ int3 make_int3(int2 a, int s) { return make_int3(a.x, a.y, s); }
inline __host__ __device__ int3 make_int3(uint3 a) { return make_int3(int(a.x), int(a.y), int(a.z)); }
inline __host__ __device__ int3 make_int3(float3 a) { return make_int3(int(a.x), int(a.y), int(a.z)); }

inline __host__ __device__ uint3 make_uint3(uint s) { return make_uint3(s, s, s); }
inline __host__ __device__ uint3 make_uint3(uint2 a) { return make_uint3(a.x, a.y, 0); }
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s) { return make_uint3(a.x, a.y, s); }
inline __host__ __device__ uint3 make_uint3(uint4 a) { return make_uint3(a.x, a.y, a.z); }
inline __host__ __device__ uint3 make_uint3(int3 a) { return make_uint3(uint(a.x), uint(a.y), uint(a.z)); }

inline __host__ __device__ float4 make_float4(float s) { return make_float4(s, s, s, s); }
inline __host__ __device__ float4 make_float4(float3 a) { return make_float4(a.x, a.y, a.z, 0.0f); }
inline __host__ __device__ float4 make_float4(float3 a, float w) { return make_float4(a.x, a.y, a.z, w); }
inline __host__ __device__ float4 make_float4(int4 a) { return make_float4(float(a.x), float(a.y), float(a.z), float(a.w)); }
inline __host__ __device__ float4 make_float4(uint4 a) { return make_float4(float(a.x), float(a.y), float(a.z), float(a.w)); }

// custom function vec4.xyz
//inline __host__ __device__ float3 fxyz(float4 a)
//{
//	return make_float3(float(a.x), float(a.y), float(a.z));
//}

inline __host__ __device__ int4 make_int4(int s) { return make_int4(s, s, s, s); }
inline __host__ __device__ int4 make_int4(int3 a) { return make_int4(a.x, a.y, a.z, 0); }
inline __host__ __device__ int4 make_int4(int3 a, int w) { return make_int4(a.x, a.y, a.z, w); }
inline __host__ __device__ int4 make_int4(uint4 a) { return make_int4(int(a.x), int(a.y), int(a.z), int(a.w)); }
inline __host__ __device__ int4 make_int4(float4 a) { return make_int4(int(a.x), int(a.y), int(a.z), int(a.w)); }

inline __host__ __device__ uint4 make_uint4(uint s) { return make_uint4(s, s, s, s); }
inline __host__ __device__ uint4 make_uint4(uint3 a) { return make_uint4(a.x, a.y, a.z, 0); }
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w) { return make_uint4(a.x, a.y, a.z, w); }
inline __host__ __device__ uint4 make_uint4(int4 a) { return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w)); }

#ifdef GLM_VERSION
inline __host__ __device__ float2 make_float2(const glm::vec2& _v) { return make_float2(_v.x, _v.y); } 
inline __host__ __device__ float3 make_float3(const glm::vec3& _v) { return make_float3(_v.x, _v.y, _v.z); } 
inline __host__ __device__ float4 make_float4(const glm::vec4& _v) { return make_float4(_v.x, _v.y, _v.z, _v.w); } 

inline __host__ __device__ glm::vec2 make_vec2(const float2& _v) { return glm::vec2(_v.x, _v.y); } 
inline __host__ __device__ glm::vec3 make_vec3(const float3& _v) { return glm::vec3(_v.x, _v.y, _v.z); } 
inline __host__ __device__ glm::vec4 mkae_vec4(const float4& _v) { return glm::vec4(_v.x, _v.y, _v.z, _v.w); } 

inline __host__ __device__ int2 make_int2(const glm::ivec2& _v) { return make_int2(_v.x, _v.y); } 
inline __host__ __device__ int3 make_int3(const glm::ivec3& _v) { return make_int3(_v.x, _v.y, _v.z); } 
inline __host__ __device__ int4 make_int4(const glm::ivec4& _v) { return make_int4(_v.x, _v.y, _v.z, _v.w); } 

inline __host__ __device__ glm::ivec2 make_ivec2(const int2& _v) { return glm::ivec2(_v.x, _v.y); } 
inline __host__ __device__ glm::ivec3 make_ivec3(const int3& _v) { return glm::ivec3(_v.x, _v.y, _v.z); } 
inline __host__ __device__ glm::ivec4 mkae_ivec4(const int4& _v) { return glm::ivec4(_v.x, _v.y, _v.z, _v.w); } 

inline __host__ __device__ uint2 make_uint2(const glm::uvec2& _v) { return make_uint2(_v.x, _v.y); } 
inline __host__ __device__ uint3 make_uint3(const glm::uvec3& _v) { return make_uint3(_v.x, _v.y, _v.z); } 
inline __host__ __device__ uint4 make_uint4(const glm::uvec4& _v) { return make_uint4(_v.x, _v.y, _v.z, _v.w); } 

inline __host__ __device__ glm::uvec2 make_uvec2(const uint2& _v) { return glm::uvec2(_v.x, _v.y); } 
inline __host__ __device__ glm::uvec3 make_uvec3(const uint3& _v) { return glm::uvec3(_v.x, _v.y, _v.z); } 
inline __host__ __device__ glm::uvec4 mkae_uvec4(const uint4& _v) { return glm::uvec4(_v.x, _v.y, _v.z, _v.w); } 
#endif

#endif

