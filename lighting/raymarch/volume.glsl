#include "map.glsl"

/*
contributors:  Shadi El Hajj
description: Default raymarching renderer. Based on Sébastien Hillaire's paper "Physically Based Sky, Atmosphere & Cloud Rendering in Frostbite"
use: <vec4> raymarchVolume( in <float3> rayOrigin, in <float3> rayDirection, in <float3> cameraForward,
    out <float3> eyeDepth ) 
options:
    - RAYMARCH_MEDIUM_DENSITY 1.0
    - LIGHT_COLOR     vec3(0.5)
    - LIGHT_INTENSITY 1.0
    - LIGHT_POSITION  vec3(0.0, 10.0, -50.0)
examples:
    - /shaders/lighting_raymarching_volume.frag
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

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND vec3(0.0)
#endif

#ifndef RAYMARCH_SAMPLES
#define RAYMARCH_SAMPLES 512
#endif

#ifndef RAYMARCH_SAMPLES
#define RAYMARCH_SAMPLES_LIGHT 8
#endif

#ifndef RAYMARCH_MIN_DIST
#define RAYMARCH_MIN_DIST 0.1
#endif

#ifndef RAYMARCH_MAX_DIST
#define RAYMARCH_MAX_DIST 10.0
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC raymarchMap
#endif

#ifndef RAYMARCH_MEDIUM_DENSITY
#define RAYMARCH_MEDIUM_DENSITY 1.0
#endif

#ifndef FNC_RAYMARCH_VOLUMERENDER
#define FNC_RAYMARCH_VOLUMERENDER

vec4 raymarchVolume( in vec3 rayOrigin, in vec3 rayDirection, vec3 cameraForward, out float eyeDepth) {

    const float tmin          = RAYMARCH_MIN_DIST;
    const float tmax          = RAYMARCH_MAX_DIST;
    const float tstep         = tmax/float(RAYMARCH_SAMPLES);
    const float tstepLight    = tmax/float(RAYMARCH_SAMPLES_LIGHT);

    #if defined(LIGHT_DIRECTION)
    vec3 lightDirection       = LIGHT_DIRECTION;
    #endif

    #if defined(LIGHT_POSITION)
    vec3 lightDirection       = normalize( LIGHT_POSITION );
    #endif

    float transmittance = 1.0;
    float t = tmin;
    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    vec3 position = rayOrigin;
    
    for(int i = 0; i < RAYMARCH_SAMPLES; i++) {
        Material res = RAYMARCH_MAP_FNC(position);
        float dist = -res.sdf;
        float density = RAYMARCH_MEDIUM_DENSITY*tstep;
        if (dist > 0.0) {
            float sampleTransmittance = exp(-dist*density);

            float transmittanceLight = 1.0;
            #if defined(LIGHT_DIRECTION) || defined(LIGHT_POSITION)
            for (int j = 0; j < RAYMARCH_SAMPLES_LIGHT; j++) {
                Material resLight = RAYMARCH_MAP_FNC(position + lightDirection * float(j) * tstepLight);
                float distLight = -resLight.sdf;
                float densityLight = RAYMARCH_MEDIUM_DENSITY*tstepLight;
                if (distLight > 0.0) {
                    transmittanceLight *= exp(-distLight*densityLight);
                }
            }
            #endif

            vec4 luminance = vec4(LIGHT_COLOR, 1.0) * LIGHT_INTENSITY * transmittanceLight;
            color += res.albedo * luminance * density * transmittance;
            transmittance *= sampleTransmittance;
        }
        position += rayDirection * tstep;
    }

    eyeDepth = t * dot(rayDirection, cameraForward);

    return color;
}

#endif
