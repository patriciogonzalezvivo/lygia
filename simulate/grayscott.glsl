#include "../math/saturate.glsl"

/*
author: Patricio Gonzalez Vivo
description: Grayscott Reaction-Diffusion
use: <vec3> grayscott(<sampler2D> tex, <vec2> st, <vec2> pixel, <float> src [, <float> diffU, <float> diffV, <float> f, <float> k ])
options:
    - GRAYSCOTT_ITERATIONS
license: |
  Copyright (c) 2022 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef GRAYSCOTT_ITERATIONS
#define GRAYSCOTT_ITERATIONS 9
#endif

#ifndef FNC_GRAYSCOTT
#define FNC_GRAYSCOTT

vec3 grayscott(sampler2D tex, vec2 st, vec2 pixel, float src, float diffU, float diffV, float f, float k ) {
    
    float kernel[9];
    kernel[0] = 0.707106781;
    kernel[1] = 1.0;
    kernel[2] = 0.707106781;
    kernel[3] = 1.0;
    kernel[4] = -6.82842712;
    kernel[5] = 1.0;
    kernel[6] = 0.707106781;
    kernel[7] = 1.0;
    kernel[8] = 0.707106781;

    vec2 offset[9];
    offset[0] = pixel * vec2(-1.0,-1.0);
    offset[1] = pixel * vec2( 0.0,-1.0);
    offset[2] = pixel * vec2( 1.0,-1.0);

    offset[3] = pixel * vec2(-1.0,0.0);
    offset[4] = pixel * vec2( 0.0,0.0);
    offset[5] = pixel * vec2( 1.0,0.0);

    offset[6] = pixel * vec2(-1.0,1.0);
    offset[7] = pixel * vec2( 0.0,1.0);
    offset[8] = pixel * vec2( 1.0,1.0);

    vec2 current = texture2D(tex, st).rb;

    vec2 lap = vec2(0.0);
    for (int i=0; i < GRAYSCOTT_ITERATIONS; i++){
        vec2 tmp = texture2D(tex, st + offset[i]).rb;
        lap += tmp * kernel[i];
    }

    float F  = f + src * 0.025 - 0.0005;
    float K  = k + src * 0.025 - 0.0005;

    float u  = current.r;
    float v  = current.g + src * 0.5;

    float uvv = u * v * v;

    float du = diffU * lap.r - uvv + F * (1.0 - u);
    float dv = diffV * lap.g + uvv - (F + K) * v;

    u += du * 0.6;
    v += dv * 0.6;
    return vec3(saturate(u), 1.0 - u/v, saturate(v));
}

vec3 grayscott(sampler2D tex, vec2 st, vec2 pixel, float src) {
    return grayscott(tex, st, pixel, src, 0.25, 0.05, 0.1, 0.063); 
}

#endif