#include "../math/const.glsl"

#ifndef RANDOM_SCALE3
#define RANDOM_SCALE3 vec3(443.897, 441.423, .0973)
#endif
#include "../generative/random.glsl"

/*
author: Alan Wolfe
description:  white noise blur based on this shader https://www.shadertoy.com/view/XsVBDR
use: noiseBlur(<sampler2D> texture, <vec2> st, <vec2> pixel, <float> radius)
options:
    - NOISEBLUR_TYPE: default to vec3
    - NOISEBLUR_GAUSSIAN_K: no gaussian by default
    - NOISEBLUR_RANDOM23_FNC(UV): defaults to random2(UV)
    - NOISEBLUR_SAMPLER_FNC(UV): defualts to texture2D(tex, UV).rgb
    - NOISEBLUR_SAMPLES: default to 4
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
licence: |
    TODO
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef NOISEBLUR_SAMPLES
#define NOISEBLUR_SAMPLES 4.0
#endif

#ifndef NOISEBLUR_TYPE
#define NOISEBLUR_TYPE vec4
#endif

#ifndef NOISEBLUR_SAMPLER_FNC
#define NOISEBLUR_SAMPLER_FNC(UV) SAMPLER_FNC(tex, UV)
#endif

#ifndef NOISEBLUR_RANDOM23_FNC
#define NOISEBLUR_RANDOM23_FNC(UV) random2(UV)
#endif

#ifndef FNC_NOISEBLUR
#define FNC_NOISEBLUR
NOISEBLUR_TYPE noiseBlur(in sampler2D tex, in vec2 st, in vec2 pixel, float radius) {
    float blurRadius = radius;
    vec2 whiteNoiseUV = st;
    NOISEBLUR_TYPE result = NOISEBLUR_TYPE(0.0);
    for (float i = 0.0; i < NOISEBLUR_SAMPLES; ++i) {
        vec2 whiteNoiseRand = NOISEBLUR_RANDOM23_FNC(vec3(whiteNoiseUV.xy, i));
        whiteNoiseUV = whiteNoiseRand;

        vec2 r = whiteNoiseRand;
        r.x *= TAU;
        
        #if defined(NOISEBLUR_GAUSSIAN_K)
        // box-muller transform to get gaussian distributed sample points in the circle
        vec2 cr = vec2(sin(r.x),cos(r.x))*sqrt(-NOISEBLUR_GAUSSIAN_K * log(r.y));
        #else
        // uniform sample the circle
        vec2 cr = vec2(sin(r.x),cos(r.x))*sqrt(r.y);
        #endif
        
        NOISEBLUR_TYPE color = NOISEBLUR_SAMPLER_FNC( st + cr * blurRadius * pixel );
        // average the samples as we get em
        // https://blog.demofox.org/2016/08/23/incremental-averaging/
        result = mix(result, color, 1.0 / (i+1.0));
    }
    return result;
}

NOISEBLUR_TYPE noiseBlur(sampler2D tex, vec2 st, vec2 pixel) {
    NOISEBLUR_TYPE rta = NOISEBLUR_TYPE(0.0);
    float total = 0.0;
    float offset = random(vec3(12.9898 + st.x, 78.233 + st.y, 151.7182));
    for (float t = -NOISEBLUR_SAMPLES; t <= NOISEBLUR_SAMPLES; t++) {
        float percent = (t / NOISEBLUR_SAMPLES) + offset - 0.5;
        float weight = 1.0 - abs(percent);
        NOISEBLUR_TYPE sample = NOISEBLUR_SAMPLER_FNC(st + pixel * percent);
        rta += sample * weight;
        total += weight;
    }
    return rta / total;
}

#endif