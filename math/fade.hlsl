/*
author: [Ian McEwan, Ashima Arts]
description: fade
use: fade(<float2|float3|float4> t)
license: |
  Copyright (C) 2011 Ashima Arts. All rights reserved.
  Distributed under the MIT License. See LICENSE file.
  https://github.com/ashima/webgl-noise
*/

#ifndef FNC_FADE
#define FNC_FADE
float2 fade(in float2 t) {
  return t * t * t * (t * (t * 6. - 15.) + 10.);
}

float3 fade(in float3 t) {
  return t * t * t * (t * (t * 6. - 15. ) + 10.);
}

float4 fade(float4 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}
#endif
