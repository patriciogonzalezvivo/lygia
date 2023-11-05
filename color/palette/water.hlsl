/*
contributors: Patricio Gonzalez Vivo
description: Simpler water color ramp 
use: <float3> water(<float> value)
*/

#ifndef FNC_WATER
#define FNC_WATER

float3 water(float x) {
    x = 4.* saturate(1.0-x);
    return pow(float3(.1, .7, .8), float3(x, x, x));
}

#endif