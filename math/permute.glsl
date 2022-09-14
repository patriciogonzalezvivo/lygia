#include "mod289.glsl"

/*
original_author: [Ian McEwan, Ashima Arts]
description: permute
use: permute(<float|vec2|vec3|vec4> x)
license : |
  Copyright (C) 2011 Ashima Arts. All rights reserved.
  Distributed under the MIT License. See LICENSE file.
  https://github.com/ashima/webgl-noise
*/

#ifndef FNC_PERMUTE
#define FNC_PERMUTE
float permute(in float x) {
     return mod289(((x * 34.) + 1.)*x);
}

vec3 permute(in vec3 x) {
  return mod289(((x*34.0)+1.0)*x);
}

vec4 permute(in vec4 x) {
     return mod289(((x * 34.) + 1.)*x);
}
#endif
