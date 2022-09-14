#include "map.glsl"

/*
original_author:  Inigo Quiles
description: calculate normals http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
use: <float> raymarchNormal( in <vec3> pos ) 
*/


#ifndef FNC_RAYMARCHNORMAL
#define FNC_RAYMARCHNORMAL

vec3 raymarchNormal(vec3 pos, float e) {
   const vec3 v1 = vec3( 1.0,-1.0,-1.0);
   const vec3 v2 = vec3(-1.0,-1.0, 1.0);
   const vec3 v3 = vec3(-1.0, 1.0,-1.0);
   const vec3 v4 = vec3( 1.0, 1.0, 1.0);

   return normalize( v1 * raymarchMap( pos + v1 * e ).a +
                     v2 * raymarchMap( pos + v2 * e ).a +
                     v3 * raymarchMap( pos + v3 * e ).a +
                     v4 * raymarchMap( pos + v4 * e ).a );
}

vec3 raymarchNormal( in vec3 pos ) {
   vec2 e = vec2(1.0, -1.0) * 0.5773 * 0.0005;
   return normalize( e.xyy * raymarchMap( pos + e.xyy ).a + 
                     e.yyx * raymarchMap( pos + e.yyx ).a + 
                     e.yxy * raymarchMap( pos + e.yxy ).a + 
                     e.xxx * raymarchMap( pos + e.xxx ).a );
}

#endif