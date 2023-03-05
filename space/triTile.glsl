/*
original_author: Mathias Bredholt
description: make some triangular tiles. XY provide coords inside of the tile. ZW provides tile coords
use: <vec4> triTile(<vec2> st [, <float> scale])
*/

#ifndef FNC_TRITILE
#define FNC_TRITILE
vec4 triTile(vec2 st) {
  st *= mat2(1., -1. / 1.7320508, 0., 2. / 1.7320508);
  vec4 f = vec4(st, -st);
  vec4 i = floor(f);
  f = fract(f);
  return dot(f.xy, f.xy) < dot(f.zw, f.zw)
             ? vec4(f.xy, vec2(2., 1.) * i.xy)
             : vec4(f.zw, -(vec2(2., 1.) * i.zw + 1.));
}

vec4 triTile(vec2 st, float scale) { return triTile(st * scale); }
#endif
