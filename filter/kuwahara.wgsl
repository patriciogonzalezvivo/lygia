#include "../sampler.wgsl"

/*
contributors: [Brad Larson, Ben Cochran, Hugues Lismonde, Keitaroh Kobayashi, Alaric Cole, Matthew Clark, Jacob Gundersen, Chris Williams.]
description: Kuwahara image abstraction, drawn from the work of Kyprianidis, et. al. in their publication "Anisotropic Kuwahara Filtering on the GPU" within the GPU Pro collection. This produces an oil-painting-like image, but it is extremely computationally expensive, so it can take seconds to render a frame on an iPad 2. This might be best used for still images.
use: kuwahara(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel, <float> radius)
options:
    - KUWAHARA_TYPE: defaults to vec3
    - KUWAHARA_SAMPLER_FNC(TEX, UV): defaults to texture2D(tex, TEX, UV).rgb
    - KUWAHARA_RADIUS radius
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
*/

// #define KUWAHARA_TYPE vec4

// #define KUWAHARA_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

KUWAHARA_TYPE kuwahara(in SAMPLER_TYPE tex, in vec2 st, in vec2 pixel, in float radius) {

//     #define KUWAHARA_RADIUS radius

    let n = (KUWAHARA_RADIUS + 1.0) * (KUWAHARA_RADIUS + 1.0);
    KUWAHARA_TYPE m0 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m1 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m2 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m3 = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE s0 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s1 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s2 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s3 = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE rta = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE c = KUWAHARA_TYPE(0.0);

    for (float j = -KUWAHARA_RADIUS; j <= 0.0; ++j)  {
        for (float i = -KUWAHARA_RADIUS; i <= 0.0; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(tex, st + vec2f(i,j) * pixel);
            m0 += c;
            s0 += c * c;
        }
    }

    for (float j = -KUWAHARA_RADIUS; j <= 0.0; ++j)  {
        for (float i = 0.0; i <= KUWAHARA_RADIUS; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(tex, st + vec2f(i,j) * pixel);
            m1 += c;
            s1 += c * c;
        }
    }

    for (float j = 0.0; j <= KUWAHARA_RADIUS; ++j)  {
        for (float i = 0.0; i <= KUWAHARA_RADIUS; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(tex, st + vec2f(i,j) * pixel);
            m2 += c;
            s2 += c * c;
        }
    }

    for (float j = 0.0; j <= KUWAHARA_RADIUS; ++j)  {
        for (float i = -KUWAHARA_RADIUS; i <= 0.0; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(tex, st + vec2f(i,j) * pixel);
            m3 += c;
            s3 += c * c;
        }
    }

    let min_sigma2 = 1e+2;
    m0 /= n;
    s0 = abs(s0 / n - m0 * m0);

    let sigma2 = s0.r + s0.g + s0.b;
    if (sigma2 < min_sigma2) {
        min_sigma2 = sigma2;
        rta = m0;
    }

    m1 /= n;
    s1 = abs(s1 / n - m1 * m1);

    sigma2 = s1.r + s1.g + s1.b;
    if (sigma2 < min_sigma2) {
        min_sigma2 = sigma2;
        rta = m1;
    }

    m2 /= n;
    s2 = abs(s2 / n - m2 * m2);

    sigma2 = s2.r + s2.g + s2.b;
    if (sigma2 < min_sigma2) {
        min_sigma2 = sigma2;
        rta = m2;
    }

    m3 /= n;
    s3 = abs(s3 / n - m3 * m3);

    sigma2 = s3.r + s3.g + s3.b;
    if (sigma2 < min_sigma2) {
        min_sigma2 = sigma2;
        rta = m3;
    }

    return rta;
}

KUWAHARA_TYPE kuwahara(in SAMPLER_TYPE tex, in vec2 st, in vec2 pixel, in float radius) {

const KUWAHARA_RADIUS: f32 = 20.0;
    let n = (radius + 1.0) * (radius + 1.0);
//     #define KUWAHARA_RADIUS radius
    let n = (KUWAHARA_RADIUS + 1.0) * (KUWAHARA_RADIUS + 1.0);

    let n = (KUWAHARA_RADIUS + 1.0) * (KUWAHARA_RADIUS + 1.0);

    KUWAHARA_TYPE m0 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m1 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m2 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m3 = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE s0 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s1 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s2 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s3 = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE rta = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE c = KUWAHARA_TYPE(0.0);
    
    for (float j = -KUWAHARA_RADIUS; j <= 0.0; ++j)  { 
        for (float i = -KUWAHARA_RADIUS; i <= 0.0; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(tex, st + vec2f(i,j) * pixel);
            m0 += c;
            s0 += c * c;
        }
    }
    
    for (float j = -KUWAHARA_RADIUS; j <= 0.0; ++j)  {
        for (float i = 0.0; i <= KUWAHARA_RADIUS; ++i)  {
            if (i > radius)
                break;
            c = KUWAHARA_SAMPLER_FNC(tex, st + vec2f(i,j) * pixel);
            m1 += c;
            s1 += c * c;
        }
    }
    
    for (float j = 0.0; j <= KUWAHARA_RADIUS; ++j)  {
        if (j > radius)
            break;
        for (float i = 0.0; i <= KUWAHARA_RADIUS; ++i)  {
            if (i > radius)
                break;
            c = KUWAHARA_SAMPLER_FNC(tex, st + vec2f(i,j) * pixel);
            m2 += c;
            s2 += c * c;
        }
    }
    
    for (float j = 0.0; j <= KUWAHARA_RADIUS; ++j)  {
        for (float i = -KUWAHARA_RADIUS; i <= 0.0; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(tex, st + vec2f(i,j) * pixel);
            m3 += c;
            s3 += c * c;
        }
    }
    
    
    let min_sigma2 = 1e+2;
    m0 /= n;
    s0 = abs(s0 / n - m0 * m0);
    
    let sigma2 = s0.r + s0.g + s0.b;
    if (sigma2 < min_sigma2) {
        min_sigma2 = sigma2;
        rta = m0;
    }
    
    m1 /= n;
    s1 = abs(s1 / n - m1 * m1);
    
    sigma2 = s1.r + s1.g + s1.b;
    if (sigma2 < min_sigma2) {
        min_sigma2 = sigma2;
        rta = m1;
    }
    
    m2 /= n;
    s2 = abs(s2 / n - m2 * m2);
    
    sigma2 = s2.r + s2.g + s2.b;
    if (sigma2 < min_sigma2) {
        min_sigma2 = sigma2;
        rta = m2;
    }
    
    m3 /= n;
    s3 = abs(s3 / n - m3 * m3);
    
    sigma2 = s3.r + s3.g + s3.b;
    if (sigma2 < min_sigma2) {
        min_sigma2 = sigma2;
        rta = m3;
    }

    return rta;
}
