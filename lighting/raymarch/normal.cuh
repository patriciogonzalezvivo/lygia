#include "../../math/make.cuh"
#include "../../math/normalize.cuh"
#include "map.cuh"

/*
contributors:  Inigo Quiles
description: Calculate normals http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
use: <float> raymarchNormal( in <float3> pos ) 
*/

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE w
#endif

#ifndef FNC_RAYMARCH_NORMAL
#define FNC_RAYMARCH_NORMAL

inline __host__ __device__ float3 raymarchNormal(const float3& pos, float2 pixel) {
   float2 offset = make_float2(1.0f, -1.0f) * pixel;
   float3 offset_xyy = make_float3(offset.x, offset.y, offset.y);
   float3 offset_yyx = make_float3(offset.y, offset.y, offset.x);
   float3 offset_yxy = make_float3(offset.y, offset.x, offset.y);
   float3 offset_xxx = make_float3(offset.x, offset.x, offset.x);
   return normalize( offset_xyy * RAYMARCH_MAP_FNC( pos + offset_xyy ).RAYMARCH_MAP_DISTANCE +
                     offset_yyx * RAYMARCH_MAP_FNC( pos + offset_yyx ).RAYMARCH_MAP_DISTANCE +
                     offset_yxy * RAYMARCH_MAP_FNC( pos + offset_yxy ).RAYMARCH_MAP_DISTANCE +
                     offset_xxx * RAYMARCH_MAP_FNC( pos + offset_xxx ).RAYMARCH_MAP_DISTANCE );
}

inline __host__ __device__ float3 raymarchNormal(const float3& pos, float e) {
   const float2 offset = make_float2(1.0f, -1.0f) * e;
   float3 offset_xyy = make_float3(offset.x, offset.y, offset.y);
   float3 offset_yyx = make_float3(offset.y, offset.y, offset.x);
   float3 offset_yxy = make_float3(offset.y, offset.x, offset.y);
   float3 offset_xxx = make_float3(offset.x, offset.x, offset.x);
   return normalize( offset_xyy * RAYMARCH_MAP_FNC( pos + offset_xyy ).RAYMARCH_MAP_DISTANCE +
                     offset_yyx * RAYMARCH_MAP_FNC( pos + offset_yyx ).RAYMARCH_MAP_DISTANCE +
                     offset_yxy * RAYMARCH_MAP_FNC( pos + offset_yxy ).RAYMARCH_MAP_DISTANCE +
                     offset_xxx * RAYMARCH_MAP_FNC( pos + offset_xxx ).RAYMARCH_MAP_DISTANCE );
}

inline __host__ __device__ float3 raymarchNormal(const float3& pos) {
    return raymarchNormal(pos, 0.5773f * 0.0005f);
}


#endif