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

#ifndef FOG_COLOR_COOL
#define FOG_COLOR_COOL float3(0.5, 0.6, 0.7)
#endif

#ifndef FOG_COLOR_WARM
#define FOG_COLOR_WARM float3(1.0, 0.9, 0.7)
#endif

#ifndef FNC_RAYMARCHFOG
#define FNC_RAYMARCHFOG

float3 raymarchFog(in float3 col, // color of pixel
                   in float t)    // distance to point
{
    float fogAmount = 1.0 - exp(-t * FOG_DENSITY);
    return lerp(col, FOG_COLOR_COOL, fogAmount);
}

float3 raymarchFog(in float3 col, // color of pixel
                in float  t,      // distance to point
                in float3 rd,     // camera to point
                in float3 lig)    // sun direction
{
    float fogAmount = 1.0 - exp(-t * FOG_DENSITY);
    float sunAmount = max(dot(rd, lig), 0.0);
    float3 fogColor = lerp(FOG_COLOR_COOL, FOG_COLOR_WARM, pow(sunAmount, 8.0));
    return lerp(col, fogColor, fogAmount);
}

#endif
