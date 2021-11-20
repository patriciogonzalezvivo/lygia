/*
author: Patricio Gonzalez Vivo
description: pass a value and get some random normalize value between 0 and 1
use: float random[2|3](<float|float2|float3> value)
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_RANDOM
#define FNC_RANDOM
float random(in float x) {
  return frac(sin(x) * 43758.5453);
}

float random(in float2 st) {
  return frac(sin(dot(st.xy, float2(12.9898, 78.233))) * 43758.5453);
}

float random(in float3 pos) {
  return frac(sin(dot(pos.xyz, float3(70.9898, 78.233, 32.4355))) * 43758.5453123);
}

float random(in float4 pos) {
    float dot_product = dot(pos, float4(12.9898, 78.233, 45.164, 94.673));
    return frac(sin(dot_product) * 43758.5453);
}

// Hash function from https://www.shadertoy.com/view/4djSRW
#define RANDOM_SCALE3 float3(.1031, .1030, .0973)
#define FANDOM_SCALE4 float4(.1031, .1030, .0973, .1099)
float2 random2(float p) {
    float3 p3 = frac(float3(p, p, p) * RANDOM_SCALE3);
    p3 += dot(p3, p3.yzx + 19.19);
    return frac((p3.xx+p3.yz)*p3.zy);
}

float2 random2(in float2 st) {
    const float2 k = float2(.3183099, .3678794);
    st = st * k + k.yx;
    return -1. + 2. * frac(16. * k * frac(st.x * st.y * (st.x + st.y)));
}

float2 random2(float3 p3) {
    p3 = frac(p3 * RANDOM_SCALE3);
    p3 += dot(p3, p3.yzx+19.19);
    return frac((p3.xx+p3.yz)*p3.zy);
}

float3 random3(float p) {
    float3 p3 = frac(float3(p, p, p) * RANDOM_SCALE3);
    p3 += dot(p3, p3.yzx+19.19);
    return frac((p3.xxy+p3.yzz)*p3.zyx); 
}

float3 random3(float2 p) {
    float3 p3 = frac(float3(p.xyx) * RANDOM_SCALE3);
    p3 += dot(p3, p3.yxz+19.19);
    return frac((p3.xxy+p3.yzz)*p3.zyx);
}

float3 random3(in float3 p) {
    p = float3( dot(p, float3(127.1, 311.7, 74.7)),
            dot(p, float3(269.5, 183.3, 246.1)),
            dot(p, float3(113.5, 271.9, 124.6)));
    return -1. + 2. * frac(sin(p) * 43758.5453123);
}

float4 random4(float p) {
    float4 p4 = frac(float4(p, p, p, p) * FANDOM_SCALE4);
    p4 += dot(p4, p4.wzxy+19.19);
    return frac((p4.xxyz+p4.yzzw)*p4.zywx);   
}

float4 random4(float2 p) {
    float4 p4 = frac(float4(p.xyxy) * FANDOM_SCALE4);
    p4 += dot(p4, p4.wzxy+19.19);
    return frac((p4.xxyz+p4.yzzw)*p4.zywx);
}

float4 random4(float3 p) {
    float4 p4 = frac(float4(p.xyzx)  * FANDOM_SCALE4);
    p4 += dot(p4, p4.wzxy+19.19);
    return frac((p4.xxyz+p4.yzzw)*p4.zywx);
}

float4 random4(float4 p4) {
    p4 = frac(p4  * FANDOM_SCALE4);
    p4 += dot(p4, p4.wzxy + 19.19);
    return frac((p4.xxyz+p4.yzzw)*p4.zywx);
}


#endif
