#include "../../math/const.hlsl"

/*
contributors:  Shadi El Hajj
description: Light Falloff equation, based on the model in Brian Karis' paper "Real Shading in Unreal Engine 4"
use: float falloff(float distance, float lightRadius)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_LIGHT_FALLOFF
#define FNC_LIGHT_FALLOFF

float falloff(float distance, float lightRadius) {
    float dr = distance/lightRadius; 
    float att = saturate(1.0 - dr*dr*dr*dr);
    att *= att;
    return att / (distance * distance + EPSILON);
}

#endif