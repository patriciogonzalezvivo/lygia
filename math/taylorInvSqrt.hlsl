/*
author: [Ian McEwan, Ashima Arts]
description: 
use: taylorInvSqrt(<float|float4> x)
License : |
  Copyright (C) 2011 Ashima Arts. All rights reserved.
  Distributed under the MIT License. See LICENSE file.
  https://github.com/ashima/webgl-noise
*/

#ifndef FNC_TAYLORINVSQRT
#define FNC_TAYLORINVSQRT
float taylorInvSqrt(in float r) {
  return 1.79284291400159 - 0.85373472095314 * r;
}

float4 taylorInvSqrt(in float4 r) {
  return 1.79284291400159 - 0.85373472095314 * r;
}
#endif