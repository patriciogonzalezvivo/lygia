#include "map.glsl"
#include "../../generative/random.glsl"
#include "../../math/const.glsl"
#include "../material/volumeNew.glsl"

/*
contributors:  Shadi El Hajj
description: Default raymarching renderer. Based on Sébastien Hillaire's paper "Physically Based Sky, Atmosphere & Cloud Rendering in Frostbite"
use: <vec4> raymarchVolume( in <vec3> rayOrigin, in <vec3> rayDirection, in <vec3> cameraForward, <vec2> st, float minDist ) 
options:
    - RAYMARCH_VOLUME_SAMPLES       256
    - RAYMARCH_VOLUME_SAMPLES_LIGHT 8
    - RAYMARCH_VOLUME_MAP_FNC       raymarchVolumeMap
    - RAYMARCH_VOLUME_DITHER        0.1
    - LIGHT_COLOR                   vec3(0.5)
    - LIGHT_INTENSITY               1.0
    - LIGHT_POSITION                vec3(0.0, 10.0, -50.0)
examples:
    - /shaders/lighting_raymarching_volume.frag
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef LIGHT_COLOR
#if defined(GLSLVIEWER)
#define LIGHT_COLOR u_lightColor
#else
#define LIGHT_COLOR vec3(0.5)
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

vec4 raymarchVolume( in vec3 rayOrigin, in vec3 rayDirection, vec2 st, float minDist) {

    const float tmin          = RAYMARCH_MIN_DIST;
    const float tmax          = RAYMARCH_MAX_DIST;
    const float tstep         = tmax/float(RAYMARCH_VOLUME_SAMPLES);
    const float tstepLight    = tmax/float(RAYMARCH_VOLUME_SAMPLES_LIGHT);

    #if defined(LIGHT_DIRECTION)
    vec3 lightDirection       = LIGHT_DIRECTION;
    #endif

    #if defined(LIGHT_POSITION)
    vec3 lightDirection       = normalize( LIGHT_POSITION );
    #endif

    float transmittance = 1.0;
    float t = tmin;
    vec3 color = vec3(0.0, 0.0, 0.0);
    vec3 position = rayOrigin;
    
    for(int i = 0; i < RAYMARCH_VOLUME_SAMPLES; i++) {
        vec3 position = rayOrigin + rayDirection * t;
        VolumeMaterial res = RAYMARCH_VOLUME_MAP_FNC(position);
        float extinction = -res.sdf;
        float density = res.density*tstep;
        if (t < minDist && extinction > 0.0) {
            float sampleTransmittance = exp(-extinction*density);

            float transmittanceLight = 1.0;
            #if defined(LIGHT_DIRECTION) || defined(LIGHT_POSITION)
            for (int j = 0; j < RAYMARCH_VOLUME_SAMPLES_LIGHT; j++) {
                VolumeMaterial resLight = RAYMARCH_VOLUME_MAP_FNC(position + lightDirection * float(j) * tstepLight);
                float extinctionLight = -resLight.sdf;
                float densityLight = res.density*tstepLight;
                if (extinctionLight > 0.0) {
                    transmittanceLight *= exp(-extinctionLight*densityLight);
                }
            }
            #endif

            vec3 luminance = LIGHT_COLOR * LIGHT_INTENSITY * transmittanceLight;

            // usual scaterring integration
            //color += res.color * luminance * density * transmittance; 
            
            // energy-conserving scattering integration
            vec3 integScatt = (luminance - luminance * sampleTransmittance) / max(extinction, EPSILON);       
            color += res.color * transmittance * integScatt;

            transmittance *= sampleTransmittance;
        }

        float offset = random(st)*(tstep*RAYMARCH_VOLUME_DITHER);
        t += tstep + offset;
    }

    return vec4(color, 1.0);
}

#endif
