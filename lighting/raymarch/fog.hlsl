/*
contributors: Inigo Quiles
description: Add fog to the scene. See https://iquilezles.org/articles/fog/
use: float3 raymarchFog(in <float3> pixelColor, float distanceToPoint)
options:
    - FOG_DENSITY
    - FOG_FALLOFF
    - FOG_COLOR_COOL
    - FOG_COLOR_WARM
*/

#ifndef FOG_DENSITY
#define FOG_DENSITY 0.0
#endif

#ifndef FOG_FALLOFF
#define FOG_FALLOFF 0.0
#endif

#ifndef FOG_COLOR_COOL
#define FOG_COLOR_COOL float3(0.5, 0.6, 0.7)
#endif

#ifndef FOG_COLOR_WARM
#define FOG_COLOR_WARM float3(1.0, 0.9, 0.7)
#endif

#ifndef FNC_RAYMARCH_FOG
#define FNC_RAYMARCH_FOG

float3 raymarchFog(in float3 col, // color of pixel
                   in float t)    // distance to point
{
    float fogAmount = 1.0 - exp(-t * FOG_DENSITY);
    return lerp(col, FOG_COLOR_COOL, fogAmount);
}

float3 raymarchColorFog(in float3 col, // color of pixel
                   in float  t,      // distance to point
                   in float3 rd,     // camera to point
                   in float3 lig)    // sun direction
{
    float fogAmount = 1.0 - exp(-t * FOG_DENSITY);
    float sunAmount = max(dot(rd, lig), 0.0);
    float3 fogColor = lerp(FOG_COLOR_COOL, FOG_COLOR_WARM, pow(sunAmount, 8.0));
    return lerp(col, fogColor, fogAmount);
}

float3 raymarchHeightFog(in float3 col, // color of pixel
                   in float t,    // distance to point
                   in float3 ro,  // camera position
                   in float3 rd)  // camera to point vector
{
    float fogAmount = (FOG_DENSITY / FOG_FALLOFF) * exp(-ro.y * FOG_FALLOFF) * (1.0 - exp(-t * rd.y * FOG_FALLOFF)) / rd.y;
    return lerp(col, FOG_COLOR_COOL, saturate(fogAmount));
}

float3 raymarchFog(in float3 col, in float t, in float3 ro, in float3 rd)
{
    if (FOG_DENSITY > 0.0 && FOG_FALLOFF > 0.0) {
        return raymarchHeightFog(col, t, ro, rd);
    }
    else if (FOG_DENSITY > 0.0) {
        #ifdef LIGHT_DIRECTION
            return raymarchColorFog(col, t, rd, LIGHT_DIRECTION);
        #else
            return raymarchFog(col, t);
        #endif
    }
    else {
        return col;
    }
}

#endif
