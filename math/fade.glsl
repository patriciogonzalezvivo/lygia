/*
author: [Ian McEwan, Ashima Arts]
description: fade
use: fade(<vec2|vec3|vec4> t)
license: |
  Copyright (C) 2011 Ashima Arts. All rights reserved.
  Distributed under the MIT License. See LICENSE file.
  https://github.com/ashima/webgl-noise
*/

#ifndef FNC_FADE
#define FNC_FADE
float fade(in float t) {
  return t * t * t * (t * (t * 6. - 15.) + 10.);
}

vec2 fade(in vec2 t) {
  return t * t * t * (t * (t * 6. - 15.) + 10.);
}

vec3 fade(in vec3 t) {
  return t * t * t * (t * (t * 6. - 15. ) + 10.);
}

vec4 fade(vec4 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}
#endif
