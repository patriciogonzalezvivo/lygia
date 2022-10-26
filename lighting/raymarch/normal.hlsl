#include "map.hlsl"

/*
original_author:  Inigo Quiles
description: calculate normals http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
use: <float> raymarchNormal( in <float3> pos ) 
*/


#ifndef FNC_RAYMARCHNORMAL
#define FNC_RAYMARCHNORMAL

float3 raymarchNormal(float3 pos, float e) {
   const float3 v1 = float3( 1.0,-1.0,-1.0);
   const float3 v2 = float3(-1.0,-1.0, 1.0);
   const float3 v3 = float3(-1.0, 1.0,-1.0);
   const float3 v4 = float3( 1.0, 1.0, 1.0);

   return normalize( v1 * raymarchMap( pos + v1 * e ).a +
                     v2 * raymarchMap( pos + v2 * e ).a +
                     v3 * raymarchMap( pos + v3 * e ).a +
                     v4 * raymarchMap( pos + v4 * e ).a );
}

float3 raymarchNormal( in float3 pos ) {
   float2 e = float2(1.0, -1.0) * 0.5773 * 0.0005;
   return normalize( e.xyy * raymarchMap( pos + e.xyy ).a + 
                     e.yyx * raymarchMap( pos + e.yyx ).a + 
                     e.yxy * raymarchMap( pos + e.yxy ).a + 
                     e.xxx * raymarchMap( pos + e.xxx ).a );
}

#endif