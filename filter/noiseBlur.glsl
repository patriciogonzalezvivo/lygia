/*
author: Alan Wolfe
description:  white noise blur based on this shader https://www.shadertoy.com/view/XsVBDR
use: noiseBlur(<sampler2D> texture, <vec2> st, <vec2> pixel, <float> radius)
options:
    NOISEBLUR_TYPE: default to vec3
    NOISEBLUR_GAUSSIAN_K: no gaussian by default
    NOISEBLUR_RANDOM23_FNC(UV): defaults to random2(UV)
    NOISEBLUR_SAMPLE_FNC(UV): defualts to texture2D(tex, UV).rgb
    NOISEBLUR_SAMPLES: default to 4
licence: |
    TODO
*/

#define RANDOM_SCALE3 vec3(443.897, 441.423, .0973)
#include "../generative/random.glsl"

#ifndef NOISEBLUR_SAMPLES
#define NOISEBLUR_SAMPLES 4
#endif

#ifndef NOISEBLUR_TYPE
#define NOISEBLUR_TYPE vec3
#endif

#ifndef NOISEBLUR_SAMPLE_FNC
#define NOISEBLUR_SAMPLE_FNC(UV) texture2D(tex, UV).rgb
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
    for (int i = 0; i < NOISEBLUR_SAMPLES; ++i) {
        vec2 whiteNoiseRand = NOISEBLUR_RANDOM23_FNC(vec3(whiteNoiseUV.xy, float(i)));
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
        
        NOISEBLUR_TYPE color = NOISEBLUR_SAMPLE_FNC( st + cr * blurRadius * pixel );
        // average the samples as we get em
        // https://blog.demofox.org/2016/08/23/incremental-averaging/
        result = mix(result, color, 1.0 / float(i+1));
    }
    return result;
}
#endif