/*
contributors: Patricio Gonzalez Vivo
description: scale a 2D space variable
use: scale(<float2> st, <float2|float> scale_factor [, <float2> center])
options:
    - CENTER
    - CENTER_2D
    - CENTER_3D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SCALE
#define FNC_SCALE
float scale(in float st, in float s, in float center) {
    return (st - center) * s + center;
}

float scale(in float st, in float s) {
    #ifdef CENTER_2D
    return scale(st,  s, CENTER);
    #else
    return scale(st,  s, .5);
    #endif
}


float2 scale(in float2 st, in float2 s, in float2 center) {
    return (st - center) * s + center;
}

float2 scale(in float2 st, in float value, in float2 center) {
    return scale(st, float2(value, value), center);
}

float2 scale(in float2 st, in float2 s) {
    #ifdef CENTER_2D
    return scale(st,  s, CENTER_2D);
    #else
    return scale(st,  s, float2(.5, .5));
    #endif
}

float2 scale(in float2 st, in float value) {
    return scale(st, float2(value, value));
}

float3 scale(in float3 st, in float3 s, in float3 center) {
    return (st - center) * s + center;
}

float3 scale(in float3 st, in float value, in float3 center) {
    return scale(st, float3(value, value, value), center);
}

float3 scale(in float3 st, in float3 s) {
    #ifdef CENTER_3D
    return scale(st,  s, CENTER_3D);
    #else
    return scale(st,  s, float3(.5, .5, .5));
    #endif
}

float3 scale(in float3 st, in float value) {
    return scale(st, float3(value, value, value));
}
#endif
