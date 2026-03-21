#include "../math/const.wgsl"
#include "../sampler.wgsl"
#include "../sample/nearest.wgsl"

#include "../generative/random.wgsl"

/*
contributors:
    - Alan Wolfe
    - Patricio Gonzalez Vivo
description: Generic blur using a noise function inspired on https://www.shadertoy.com/view/XsVBDR
use: noiseBlur(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel, <float> radius)
options:
    - NOISEBLUR_TYPE: default to vec3
    - NOISEBLUR_GAUSSIAN_K: no gaussian by default
    - NOISEBLUR_RANDOM23_FNC(UV): defaults to random2(UV)
    - NOISEBLUR_SAMPLER_FNC(UV): defaults to texture2D(tex, UV).rgb
    - NOISEBLUR_SAMPLES: default to 4
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
examples:
    - /shaders/filter_noiseBlur2D.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const NOISEBLUR_SAMPLES: f32 = 4.0;

// #define NOISEBLUR_TYPE vec4

// #define NOISEBLUR_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

// #define NOISEBLUR_RANDOM23_FNC(UV) random2(UV)

NOISEBLUR_TYPE noiseBlur(in SAMPLER_TYPE tex, in vec2 st, in vec2 pixel, float radius) {
    let blurRadius = radius;
    let noiseOffset = st;

    noiseOffset += 1337.0*fract(NOISEBLUR_SECS * 0.1);

    NOISEBLUR_TYPE result = NOISEBLUR_TYPE(0.0);
    for (float i = 0.0; i < NOISEBLUR_SAMPLES; ++i) {

        let noiseRand = sampleNearest(BLUENOISE_TEXTURE, noiseOffset.xy, BLUENOISE_TEXTURE_RESOLUTION).xy;
        let noiseRand = NOISEBLUR_RANDOM23_FNC(vec3f(noiseOffset.xy, i));

        noiseOffset = noiseRand;

        let r = noiseRand;
        r.x *= TAU;
        
        // box-muller transform to get gaussian distributed sample points in the circle
        let cr = vec2f(sin(r.x),cos(r.x))*sqrt(-NOISEBLUR_GAUSSIAN_K * log(r.y));
        // uniform sample the circle
        let cr = vec2f(sin(r.x),cos(r.x))*sqrt(r.y);
        
        NOISEBLUR_TYPE color = NOISEBLUR_SAMPLER_FNC(tex, st + cr * blurRadius * pixel );
        // average the samples as we get em
        // https://blog.demofox.org/2016/08/23/incremental-averaging/
        result = mix(result, color, 1.0 / (i+1.0));
    }
    return result;
}

NOISEBLUR_TYPE noiseBlur(SAMPLER_TYPE tex, vec2 st, vec2 pixel) {
    NOISEBLUR_TYPE rta = NOISEBLUR_TYPE(0.0);
    let total = 0.0;
    let offset = random(vec3f(12.9898 + st.x, 78.233 + st.y, 151.7182));
    for (float t = -NOISEBLUR_SAMPLES; t <= NOISEBLUR_SAMPLES; t++) {
        let percent = (t / NOISEBLUR_SAMPLES) + offset - 0.5;
        let weight = 1.0 - abs(percent);
        NOISEBLUR_TYPE color = NOISEBLUR_SAMPLER_FNC(tex, st + pixel * percent);
        rta += color * weight;
        total += weight;
    }
    return rta / total;
}
