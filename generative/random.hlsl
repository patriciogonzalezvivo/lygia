/*
contributors: ["Patricio Gonzalez Vivo", "David Hoskins", "Inigo Quilez"]
description: Pass a value and get some random normalize value between 0 and 1
use: float random[2|3](<float|float2|float3> value)
options:
    - RANDOM_HIGHER_RANGE: for working with a range over 0 and 1
    - RANDOM_SINLESS: Use sin-less random, which tolerates bigger values before producing pattern. From https://www.shadertoy.com/view/4djSRW
    - RANDOM_SCALE: by default this scale if for number with a big range. For producing good random between 0 and 1 use bigger range
examples:
    - /shaders/generative_random.frag
license:
    - MIT License (MIT) Copyright 2014, David Hoskins
*/

#ifndef RANDOM_SCALE
#if defined(RANDOM_HIGHER_RANGE)
#define RANDOM_SCALE float4(0.1031, 0.1030, 0.0973, 0.1099)
#else
#define RANDOM_SCALE float4(443.897, 441.423, 0.0973, 0.1099)
#endif
#endif
#ifndef RANDOM_SCALE_4
#define RANDOM_SCALE_4 float4(443.897, 441.423, .0973, 1.6334)
#endif

#ifndef FNC_RANDOM
#define FNC_RANDOM

float random(float x)
{
    #if defined(RANDOM_SINLESS)
    x = frac(x * RANDOM_SCALE.x);
    x *= x + 33.33;
    x *= x + x;
    return frac(x);
    #else
    return frac(sin(x) * 43758.5453);
    #endif
}

float random(float2 st)
{
    #if defined(RANDOM_SINLESS)
    float3 p3  = frac(st.xx * RANDOM_SCALE.xyz);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
    #else
    return frac(sin(dot(st.xy, float2(12.9898, 78.233))) * 43758.5453);
    #endif
}

float random(float3 pos)
{
    #if defined(RANDOM_SINLESS)
    pos  = frac(pos * RANDOM_SCALE.xyz);
    pos += dot(pos, pos.zyx + 31.32);
    return frac((pos.x + pos.y) * pos.z);
    #else
    return frac(sin(dot(pos.xyz, float3(70.9898, 78.233, 32.4355))) * 43758.5453123);
    #endif
}

float random(float4 pos)
{
    #if defined(RANDOM_SINLESS)
    pos = frac(pos * RANDOM_SCALE);
    pos += dot(pos, pos.wzyx + 33.33);
    return frac((pos.x + pos.y) * (pos.z + pos.w));
    #else
    return frac(sin(dot(pos, float4(12.9898, 78.233, 45.164, 94.673))) * 43758.5453);
    #endif
}

float2 random2(float p)
{
    float3 p3 = frac(float3(p, p, p) * RANDOM_SCALE.xyz);
    p3 += dot(p3, p3.yzx + 19.19);
    return frac((p3.xx + p3.yz) * p3.zy);
}

float2 random2(float2 p)
{
    float3 p3 = frac(float3(p.x, p.y, p.x) * RANDOM_SCALE.xyz);
    p3 += dot(p3, p3.yzx + 19.19);
    return frac((p3.xx + p3.yz) * p3.zy);
}

float2 random2(float3 p3)
{
    p3 = frac(p3 * RANDOM_SCALE.xyz);
    p3 += dot(p3, p3.yzx + 19.19);
    return frac((p3.xx + p3.yz) * p3.zy);
}

float3 random3(float p)
{
    float3 scale = float3(RANDOM_SCALE.x, RANDOM_SCALE.y, RANDOM_SCALE.z);
    float3 p3 = frac(float3(p, p, p) * scale);
    p3 += dot(p3, p3.yzx + 19.19);
    return frac((p3.xxy + p3.yzz) * p3.zyx);
}


float3 random3(float2 p)
{
    float3 scale = float3(RANDOM_SCALE.x, RANDOM_SCALE.y, RANDOM_SCALE.z);
    float3 p3 = frac(float3(p.x, p.x, p.x) * scale);
    p3 += dot(p3, p3.yzx + 19.19);
    return frac((p3.xxy + p3.yzz) * p3.zyx);
}


float3 random3(float3 p)
{
    p = frac(p * RANDOM_SCALE.xyz);
    p += dot(p, p.yzx + 19.19);
    return frac((p.xxy + p.yzz) * p.zyx);
}

float4 random4(float p)
{
    float4 p4 = frac(float4(p, p, p, p) * RANDOM_SCALE_4);
    p4 += dot(p4, p4.wzyx + 19.19);
    return frac((p4.xxyz + p4.yzzw) * p4.zywx);
}

float4 random4(float2 p)
{
    float4 p4 = frac(float4(p.xx, p.yy) * RANDOM_SCALE_4);
    p4 += dot(p4, p4.wzyx + 19.19);
    return frac((p4.xxyz + p4.yzzw) * p4.zywx);
}

float4 random4(float3 p)
{
    float4 p4 = frac(float4(p.xyz, 1.0) * RANDOM_SCALE_4);
    p4 += dot(p4, p4.wzyx + 19.19);
    return frac((p4.xxyz + p4.yzzw) * p4.zywx);
}

float4 random4(float4 p4)
{
    p4 = frac(p4 * RANDOM_SCALE_4);
    p4 += dot(p4, p4.wzyx + 19.19);
    return frac((p4.xxyz + p4.yzzw) * p4.zywx);
}

#endif
