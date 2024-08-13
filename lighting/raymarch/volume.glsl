#include "map.glsl"
#include "normal.glsl"

/*
contributors:  Inigo Quiles
description: Default raymarching renderer
use: <vec4> raymarchVolume( in <float3> rayOriging, in <float3> rayDirection, in <float3> cameraForward,
    out <float3> eyeDepth, out <float3> worldPosition, out <float3> worldNormal ) 
options:
    - RAYMARCH_BACKGROUND vec3(0.0)
    - RAYMARCH_AMBIENT vec3(1.0)
    - LIGHT_COLOR     vec3(0.5)
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
#define LIGHT_INTENSITY 100.0
#endif

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND vec3(0.0)
#endif

#ifndef RAYMARCH_SAMPLES
#define RAYMARCH_SAMPLES 512
#endif

#ifndef RAYMARCH_MIN_DIST
#define RAYMARCH_MIN_DIST 0.1
#endif

#ifndef RAYMARCH_MAX_DIST
#define RAYMARCH_MAX_DIST 20.0
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC raymarchMap
#endif

#ifndef RAYMARCH_MEDIUM_DENSITY
#define RAYMARCH_MEDIUM_DENSITY 100.0
#endif

#ifndef FNC_RAYMARCH_VOLUMERENDER
#define FNC_RAYMARCH_VOLUMERENDER

vec4 raymarchVolume( in vec3 rayOrigin, in vec3 rayDirection, vec3 cameraForward,
                     out float eyeDepth, out vec3 worldPos, out vec3 worldNormal) {

    const float tmin        = RAYMARCH_MIN_DIST;
    const float tmax        = RAYMARCH_MAX_DIST;
    const float fSamples    = float(RAYMARCH_SAMPLES);
    const float tstep       = tmax/fSamples;
    const float mediumDensity = RAYMARCH_MEDIUM_DENSITY / fSamples;

    #ifdef LIGHT_POSITION
    const int   nbSampleLight   = 6;
    const float fSampleLight    = float(nbSampleLight);
    const float tstepl          = tmax/fSampleLight;
    vec3 sun_direction          = normalize( LIGHT_POSITION );
    #endif

    float transmittance = 1.;
    float t = tmin;
    vec4 col = vec4(0.0, 0.0, 0.0, 0.0);
    vec3 pos = rayOrigin;
    for(int i = 0; i < RAYMARCH_SAMPLES; i++) {
        Material res = RAYMARCH_MAP_FNC(pos);
        float dist = -res.sdf;
        if (dist > 0.0) {
            float density = saturate(dist * mediumDensity);
            transmittance *= 1.0 - density;

            col += res.albedo * density * transmittance;

            //Light scattering
            #ifdef LIGHT_POSITION
            float transmittanceLight = 1.0;
            for (int j = 0; j < nbSampleLight; j++) {
                Material resLight = RAYMARCH_MAP_FNC( pos + sun_direction * float(j) * tstepl );
                float distLight = resLight.sdf;
                if (distLight > 0.0) {
                    float densityLight = saturate(distLight * mediumDensity);
                    transmittanceLight *= 1.0 - densityLight;
                }
            }
            col += vec4(LIGHT_COLOR, 1.0) * LIGHT_INTENSITY * dist / fSamples * transmittance * transmittanceLight;
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