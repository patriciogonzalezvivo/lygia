/*
contributors: Patricio Gonzalez Vivo
description: Simpler fire color ramp 
use: <float3> fire(<float> value)
*/

#ifndef FNC_FIRE
#define FNC_FIRE
float3 fire(float x) { return float3(1.0, 0.25, 0.0625) * exp(4.0 * x - 1.0); }
#endif