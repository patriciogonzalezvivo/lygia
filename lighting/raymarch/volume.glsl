#include "../../math/saturate.glsl"

/*
contributors:  Inigo Quiles
description: default raymarching renderer
use: <vec4> raymarchDefaultRender( in <vec3> ro, in <vec3> rd ) 
options:
    - RAYMARCH_MATERIAL_FNC(RGB) vec3(RGB)
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

#ifndef RAYMARCH_AMBIENT
#define RAYMARCH_AMBIENT vec3(1.0)
#endif

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND vec3(0.0)
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

#ifndef RAYMARCH_VOLUME_COLOR_FNC
#define RAYMARCH_VOLUME_COLOR_FNC vec3
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef FNC_RAYMARCHVOLUMERENDER
#define FNC_RAYMARCHVOLUMERENDER

vec4 raymarchVolume( in vec3 ro, in vec3 rd ) {

    const float tmin        = RAYMARCH_MIN_DIST;
    const float tmax        = RAYMARCH_MAX_DIST;
    const float fSamples    = float(RAYMARCH_SAMPLES);
    const float tstep       = tmax/fSamples;
    const float absorption  = 100.;

    #ifdef LIGHT_POSITION
    const int   nbSampleLight   = 6;
    const float fSampleLight    = float(nbSampleLight);
    const float tstepl          = tmax/fSampleLight;
    vec3 sun_direction          = normalize( LIGHT_POSITION );
    #endif

    float T = 1.;
    float t = tmin;
    vec3 col = vec3(0.0);
    vec3 pos = ro;
    for(int i = 0; i < RAYMARCH_SAMPLES; i++) {
        vec4 res    = RAYMARCH_MAP_FNC(pos);
        float density = (0.1 - res.a);
        if (density > 0.0) {
            float tmp = density / fSamples;
            T *= 1.0 - tmp * absorption;
            if( T <= 0.001)
                break;

            col += RAYMARCH_VOLUME_COLOR_FNC(res.rgb) * fSamples * tmp * T;
                
            //Light scattering
            #ifdef LIGHT_POSITION
            float Tl = 1.0;
            for (int j = 0; j < nbSampleLight; j++) {
                float densityLight = raymarchMap( pos + sun_direction * float(j) * tstepl ).a;
                if (densityLight>0.)
                    Tl *= 1. - densityLight * absorption/fSamples;
                if (Tl <= 0.01)
                    break;
            }
            col += LIGHT_COLOR * 80. * tmp * T * Tl;
            #endif
        }
        pos += rd * tstep;
    }

    return vec4(saturate(col), t);
}

#endif