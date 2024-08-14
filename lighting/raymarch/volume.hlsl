#include "map.hlsl"
#include "normal.hlsl"

/*
contributors:  Inigo Quiles
description: Default raymarching renderer
use: <float4> raymarchVolume( in <float3> rayOriging, in <float3> rayDirection, in <float3> cameraForward,
    out <float3> eyeDepth, out <float3> worldPosition, out <float3> worldNormal ) 
options:
    - RAYMARCH_BACKGROUND float3(0.0)
    - LIGHT_COLOR     float3(0.5)
    - LIGHT_POSITION  float3(0.0, 10.0, -50.0)
*/

#ifndef LIGHT_COLOR
#if defined(GLSLVIEWER)
#define LIGHT_COLOR u_lightColor
#else
#define LIGHT_COLOR float3(0.5, 0.5, 0.5)
#endif
#endif

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND float3(0.0, 0.0, 0.0)
#endif

#ifndef RAYMARCH_SAMPLES
#define RAYMARCH_SAMPLES 64
#endif

#ifndef RAYMARCH_MIN_DIST
#define RAYMARCH_MIN_DIST 1.0
#endif

#ifndef RAYMARCH_MAX_DIST
#define RAYMARCH_MAX_DIST 10.0
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef FNC_RAYMARCH_VOLUMERENDER
#define FNC_RAYMARCH_VOLUMERENDER

float4 raymarchVolume(in float3 rayOrigin, in float3 rayDirection, float3 cameraForward,
                      out float eyeDepth, out float3 worldPos, out float3 worldNormal)
{

    const float tmin        = RAYMARCH_MIN_DIST;
    const float tmax        = RAYMARCH_MAX_DIST;
    const float fSamples    = float(RAYMARCH_SAMPLES);
    const float tstep       = tmax/fSamples;
    const float absorption  = 100.;

    #ifdef LIGHT_POSITION
    const int   nbSampleLight   = 6;
    const float fSampleLight    = float(nbSampleLight);
    const float tstepl          = tmax/fSampleLight;
    float3 sun_direction          = normalize( LIGHT_POSITION );
    #endif

    float T = 1.;
    float t = tmin;
    float4 col = float4(0.0, 0.0, 0.0, 0.0);
    float3 pos = rayOrigin;
    for(int i = 0; i < RAYMARCH_SAMPLES; i++) {
        Material res    = RAYMARCH_MAP_FNC(pos);
        float density = (0.1 - res.sdf);
        if (density > 0.0) {
            float tmp = density / fSamples;
            T *= 1.0 - tmp * absorption;
            if( T <= 0.001)
                break;

            col += res.albedo * fSamples * tmp * T;
                
            //Light scattering
            #ifdef LIGHT_POSITION
            float Tl = 1.0;
            for (int j = 0; j < nbSampleLight; j++) {
                float densityLight = RAYMARCH_MAP_FNC( pos + sun_direction * float(j) * tstepl ).sdf;
                if (densityLight>0.)
                    Tl *= 1. - densityLight * absorption/fSamples;
                if (Tl <= 0.01)
                    break;
            }
            col += float4(LIGHT_COLOR * 80. * tmp * T * Tl, 1.0);
            #endif
        }
        pos += rayDirection * tstep;
    }

    worldPos = rayOrigin + t * rayDirection;
    worldNormal = raymarchNormal( worldPos );
    eyeDepth = t * dot(rayDirection, cameraForward);

    return col;
}

#endif