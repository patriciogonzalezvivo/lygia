#include "map.glsl"

/*
contributors:  Inigo Quiles
description: Calculate normals http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
use: <float> raymarchNormal( in <vec3> pos ) 
examples:
    - /shaders/lighting_raymarching.frag
*/

#ifndef RAYMARCH_NORMAL_OFFSET
#define RAYMARCH_NORMAL_OFFSET 0.0001
#endif

#ifndef FNC_RAYMARCH_NORMAL
#define FNC_RAYMARCH_NORMAL

vec3 raymarchNormal(vec3 pos, vec2 pixel) {
   vec2 offset = vec2(1.0, -1.0) * pixel;
   return normalize( offset.xyy * RAYMARCH_MAP_FNC( pos + offset.xyy ).sdf +
                     offset.yyx * RAYMARCH_MAP_FNC( pos + offset.yyx ).sdf +
                     offset.yxy * RAYMARCH_MAP_FNC( pos + offset.yxy ).sdf +
                     offset.xxx * RAYMARCH_MAP_FNC( pos + offset.xxx ).sdf );
}

vec3 raymarchNormal(vec3 pos, float e) {
   vec2 offset = vec2(1.0, -1.0) * e;
   return normalize( offset.xyy * RAYMARCH_MAP_FNC( pos + offset.xyy ).sdf +
                     offset.yyx * RAYMARCH_MAP_FNC( pos + offset.yyx ).sdf +
                     offset.yxy * RAYMARCH_MAP_FNC( pos + offset.yxy ).sdf +
                     offset.xxx * RAYMARCH_MAP_FNC( pos + offset.xxx ).sdf );
}

vec3 raymarchNormal( in vec3 pos ) {
   return raymarchNormal(pos, RAYMARCH_NORMAL_OFFSET);
}

#endif