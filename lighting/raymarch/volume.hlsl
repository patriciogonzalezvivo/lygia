#include "map.hlsl"
#include "../common/attenuation.hlsl"
#include "../../generative/random.hlsl"
#include "../../math/const.hlsl"
#include "../material/volumeNew.hlsl"

/*
contributors:  Shadi El Hajj
description: Default raymarching renderer. Based on Sébastien Hillaire's paper "Physically Based Sky, Atmosphere & Cloud Rendering in Frostbite"
use: <float4> raymarchVolume( in <float3> rayOrigin, in <float3> rayDirection, in <float3> cameraForward, <float2> st, float minDist ) 
options:
    - RAYMARCH_VOLUME_SAMPLES       256
    - RAYMARCH_VOLUME_SAMPLES_LIGHT 8
    - RAYMARCH_VOLUME_MAP_FNC       raymarchVolumeMap
    - RAYMARCH_VOLUME_DITHER        0.1
    - RAYMARCH_ENERGY_CONSERVING
    - LIGHT_COLOR                   float3(0.5, 0.5, 0.5)
    - LIGHT_INTENSITY               1.0
    - LIGHT_POSITION
    - LIGHT_DIRECTION
examples:
    - /shaders/lighting_raymarching_volume.frag
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef LIGHT_COLOR
#if defined(hlslVIEWER)
#define LIGHT_COLOR u_lightColor
#else
#define LIGHT_COLOR float3(0.5, 0.5, 0.5)
#endif
#endif

#ifndef LIGHT_INTENSITY
#define LIGHT_INTENSITY 1.0
#endif

#ifndef RAYMARCH_VOLUME_SAMPLES
#define RAYMARCH_VOLUME_SAMPLES 256
#endif

#ifndef RAYMARCH_VOLUME_SAMPLES_LIGHT
#define RAYMARCH_VOLUME_SAMPLES_LIGHT 8
#endif

#ifndef RAYMARCH_VOLUME_MAP_FNC
#define RAYMARCH_VOLUME_MAP_FNC raymarchVolumeMap
#endif

#ifndef RAYMARCH_VOLUME_DITHER
#define RAYMARCH_VOLUME_DITHER 0.1
#endif

#ifndef FNC_RAYMARCH_VOLUMERENDER
#define FNC_RAYMARCH_VOLUMERENDER

float3 shadowTransmittance(float3 position, float3 rayDirectionL, float stepSizeL) {
    float3 transmittanceL = float3(1.0, 1.0, 1.0);
    
    for (int j = 0; j < RAYMARCH_VOLUME_SAMPLES_LIGHT; j++) {                
        float3 positionL = position + rayDirectionL * j * stepSizeL;
        VolumeMaterial resL = RAYMARCH_VOLUME_MAP_FNC(positionL);
        float densityL = -resL.sdf;
        float3 extinctionL = resL.absorption + resL.scattering;
        transmittanceL *= exp(-densityL * extinctionL * stepSizeL);
    }

    return transmittanceL;
}

float3 raymarchVolume( in float3 rayOrigin, in float3 rayDirection, float2 st, float minDist, float3 background) {
    float3 scatteredLuminance = float3(0.0, 0.0, 0.0);
    float3 transmittance = float3(1.0, 1.0, 1.0);
    float stepSize = RAYMARCH_MAX_DIST/float(RAYMARCH_VOLUME_SAMPLES);

    float t = RAYMARCH_MIN_DIST;
    float3 position = rayOrigin;

    for(int i = 0; i < RAYMARCH_VOLUME_SAMPLES; i++) {        
        float3 position = rayOrigin + rayDirection * t;
        VolumeMaterial res = RAYMARCH_VOLUME_MAP_FNC(position);
        float density = -res.sdf;
        float3 extinction = res.absorption + res.scattering;

        if (t < minDist && density > 0.0) {

            #if defined(LIGHT_DIRECTION) || defined(LIGHT_POSITION)
                #if defined(LIGHT_DIRECTION) // directional light
                    float stepSizeL = RAYMARCH_MAX_DIST/float(RAYMARCH_VOLUME_SAMPLES_LIGHT);
                    float3 rayDirectionL = LIGHT_DIRECTION;
                    const float attenuationL = 1.0;
                #else // point light
                    float distL = distance(LIGHT_POSITION, position);
                    float stepSizeL = distL/float(RAYMARCH_VOLUME_SAMPLES_LIGHT);
                    float3 rayDirectionL = normalize(LIGHT_POSITION - position);
                    float attenuationL = attenuation(distL);
                #endif
                float3 shadow = shadowTransmittance(position, rayDirectionL, stepSizeL);
                float3 L = LIGHT_COLOR * LIGHT_INTENSITY;
            #else // no lighting
                const float attenuationL = 1.0;
                const float3 shadow = 1.0;
                const float3 L = float3(1.0, 1.0, 1.0);
            #endif

            #if defined RAYMARCH_ENERGY_CONSERVING
                // energy-conserving scattering integration
                float3 S = L * attenuationL * shadow * density * res.scattering;
                float3 sampleExtinction = max(float3(EPSILON, EPSILON, EPSILON), density * extinction);
                float3 Sint = (S - S * exp(-sampleExtinction * stepSize)) / sampleExtinction;
                scatteredLuminance += transmittance * Sint;
                transmittance *= exp(-sampleExtinction * stepSize);
            #else
                // usual scattering integration. Not energy-conserving.
                scatteredLuminance += attenuationL * shadow * transmittance * density * res.scattering * stepSize * L;
                transmittance *= exp(-density * extinction * stepSize);
            #endif
        }

        float offset = random(st)*(stepSize*RAYMARCH_VOLUME_DITHER);
        t += stepSize + offset;
    }

    return background * transmittance + scatteredLuminance;
}

#endif
