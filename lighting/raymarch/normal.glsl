#include "map.glsl"

/*
contributors:  Inigo Quiles
description: calculate normals http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
use: <float> raymarchNormal( in <vec3> pos ) 
examples:
    - /shaders/lighting_raymarching.frag
*/

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE a
#endif

#ifndef FNC_RAYMARCHNORMAL
#define FNC_RAYMARCHNORMAL

vec3 raymarchNormal(vec3 pos, vec2 pixel) {
   vec2 offset = vec2(1.0, -1.0) * pixel;
   return normalize( offset.xyy * RAYMARCH_MAP_FNC( pos + offset.xyy ).RAYMARCH_MAP_DISTANCE +
                     offset.yyx * RAYMARCH_MAP_FNC( pos + offset.yyx ).RAYMARCH_MAP_DISTANCE +
                     offset.yxy * RAYMARCH_MAP_FNC( pos + offset.yxy ).RAYMARCH_MAP_DISTANCE +
                     offset.xxx * RAYMARCH_MAP_FNC( pos + offset.xxx ).RAYMARCH_MAP_DISTANCE );
}

vec3 raymarchNormal(vec3 pos, float e) {
   vec2 offset = vec2(1.0, -1.0) * e;
   return normalize( offset.xyy * RAYMARCH_MAP_FNC( pos + offset.xyy ).RAYMARCH_MAP_DISTANCE +
                     offset.yyx * RAYMARCH_MAP_FNC( pos + offset.yyx ).RAYMARCH_MAP_DISTANCE +
                     offset.yxy * RAYMARCH_MAP_FNC( pos + offset.yxy ).RAYMARCH_MAP_DISTANCE +
                     offset.xxx * RAYMARCH_MAP_FNC( pos + offset.xxx ).RAYMARCH_MAP_DISTANCE );
}

vec3 raymarchNormal( in vec3 pos ) {
   return raymarchNormal(pos, 0.5773 * 0.0005);
}

#endif