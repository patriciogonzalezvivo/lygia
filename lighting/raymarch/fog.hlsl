/*
contributors:  Inigo Quiles
description: Add fog to the scene. See https://iquilezles.org/articles/fog/
use: float3 raymarchFog(in <float3> pixelColor, float distanceToPoint)
options:
    - FOG_DENSITY
    - FOG_COLOR
*/

#ifndef FOG_DENSITY
#define FOG_DENSITY 0.0
#endif

#ifndef FOG_COLOR
#define FOG_COLOR float3(0.5, 0.6, 0.7)
#endif

#ifndef FNC_RAYMARCHFOG
#define FNC_RAYMARCHFOG

float3 fog(in float3 col, in float t)
{
    float fogAmount = 1.0 - exp(-t * FOG_DENSITY);
    return lerp(col, FOG_COLOR, fogAmount);
}

#endif