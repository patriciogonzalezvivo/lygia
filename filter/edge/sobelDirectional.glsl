/*
author: Brad Larson
description: Adapted version of directional Sobel edge detection from https://github.com/BradLarson/GPUImage2
use: edgeSobel_directional(<sampler2D> texture, <vec2> st, <vec2> pixels_scale)
options:
  EDGESOBEL_DIRECTIONAL_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,POS_UV).r
license: |
  Copyright (c) 2015, Brad Larson.
  All rights reserved.
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met

  Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef EDGESOBEL_DIRECTIONAL_SAMPLER_FNC
#define EDGESOBEL_DIRECTIONAL_SAMPLER_FNC(POS_UV) texture2D(tex,POS_UV).r
#endif

#ifndef FNC_EDGESOBEL_DIRECTIONAL
#define FNC_EDGESOBEL_DIRECTIONAL
vec3 edgeSobelDirectional(in sampler2D tex, in vec2 st, in vec2 offset) {
    // get samples around pixel
    float tleft = EDGESOBEL_DIRECTIONAL_SAMPLER_FNC(st + vec2(-offset.x, offset.y));
    float left = EDGESOBEL_DIRECTIONAL_SAMPLER_FNC(st + vec2(-offset.x, 0.));
    float bleft = EDGESOBEL_DIRECTIONAL_SAMPLER_FNC(st + vec2(-offset.x, -offset.y));
    float top = EDGESOBEL_DIRECTIONAL_SAMPLER_FNC(st + vec2(0., offset.y));
    float bottom = EDGESOBEL_DIRECTIONAL_SAMPLER_FNC(st + vec2(0., -offset.y));
    float tright = EDGESOBEL_DIRECTIONAL_SAMPLER_FNC(st + offset);
    float right = EDGESOBEL_DIRECTIONAL_SAMPLER_FNC(st + vec2(offset.x, 0.));
    float bright = EDGESOBEL_DIRECTIONAL_SAMPLER_FNC(st + vec2(offset.x, -offset.y));
    vec2 gradientDirection = vec2(0.);
    gradientDirection.x = -bleft - 2. * left - tleft + bright + 2. * right + tright;
    gradientDirection.y = -tleft - 2. * top - tright + bleft + 2. * bottom + bright;
    float gradientMagnitude = length(gradientDirection);
    vec2 normalizedDirection = normalize(gradientDirection);
    normalizedDirection = sign(normalizedDirection) * floor(abs(normalizedDirection) + .617316); // Offset by 1-sin(pi/8) to set to 0 if near axis, 1 if away
    normalizedDirection = (normalizedDirection + 1.) * .5; // Place -1.0 - 1.0 within 0 - 1.0
    return vec3(gradientMagnitude, normalizedDirection.x, normalizedDirection.y);
}
#endif
