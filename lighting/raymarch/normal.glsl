#include "map.glsl"

/*
original_author:  Inigo Quiles
description: calculate normals http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
use: <float> raymarchNormal( in <vec3> pos ) 
*/


#ifndef FNC_RAYMARCHNORMAL
#define FNC_RAYMARCHNORMAL

vec3 raymarchNormal(vec3 pos, vec2 pixel) {
   vec2 offset = vec2(1.0, -1.0) * pixel;
   return normalize( offset.xyy * raymarchMap( pos + offset.xyy ).a +
                     offset.yyx * raymarchMap( pos + offset.yyx ).a +
                     offset.yxy * raymarchMap( pos + offset.yxy ).a +
                     offset.xxx * raymarchMap( pos + offset.xxx ).a );
}

vec3 raymarchNormal(vec3 pos, float e) {
   const vec2 offset = vec2(1.0, -1.0);
   return normalize( offset.xyy * raymarchMap( pos + offset.xyy * e ).a +
                     offset.yyx * raymarchMap( pos + offset.yyx * e ).a +
                     offset.yxy * raymarchMap( pos + offset.yxy * e ).a +
                     offset.xxx * raymarchMap( pos + offset.xxx * e ).a );
}

vec3 raymarchNormal( in vec3 pos ) {
   return raymarchNormal(pos, 0.5773 * 0.0005);
}

#endif