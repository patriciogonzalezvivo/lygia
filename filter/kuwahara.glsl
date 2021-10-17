/*
author: Brad Larson
description: Kuwahara image abstraction, drawn from the work of Kyprianidis, et. al. in their publication "Anisotropic Kuwahara Filtering on the GPU" within the GPU Pro collection. This produces an oil-painting-like image, but it is extremely computationally expensive, so it can take seconds to render a frame on an iPad 2. This might be best used for still images.
use: kuwahara(<sampler2D> texture, <vec2> st, <vec2> pixel)
options:
    KUWAHARA_TYPE: defaults to vec3
    KUWAHARA_SAMPLER_FNC(POS_UV): defaults to texture2D(tex, POS_UV).rgb
    KUWAHARA_RADIUS radius
licence:
    Copyright (c) 2012, Brad Larson, Ben Cochran, Hugues Lismonde, Keitaroh Kobayashi, Alaric Cole, Matthew Clark, Jacob Gundersen, Chris Williams.
    All rights reserved.
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    Neither the name of the GPUImage framework nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef KUWAHARA_TYPE
#define KUWAHARA_TYPE vec3
#endif

#ifndef KUWAHARA_SAMPLER_FNC
#define KUWAHARA_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV).rgb
#endif

#ifndef KUWAHARA_RADIUS
#define KUWAHARA_RADIUS radius
#endif

#ifndef FNC_KUWAHARA
#define FNC_KUWAHARA

#ifdef TARGET_MOBILE
KUWAHARA_TYPE kuwahara(in sampler2D tex, in vec2 st, in vec2 pixel, in int radius) {
    float n = float((KUWAHARA_RADIUS + 1) * (KUWAHARA_RADIUS + 1));
    int i; int j;
    KUWAHARA_TYPE m0 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m1 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m2 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m3 = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE s0 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s1 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s2 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s3 = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE rta = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE c;

    for (j = -KUWAHARA_RADIUS; j <= 0; ++j)  {
        for (i = -KUWAHARA_RADIUS; i <= 0; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(st + vec2(i,j) * pixel);
            m0 += c;
            s0 += c * c;
        }
    }

    for (j = -KUWAHARA_RADIUS; j <= 0; ++j)  {
        for (i = 0; i <= KUWAHARA_RADIUS; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(st + vec2(i,j) * pixel);
            m1 += c;
            s1 += c * c;
        }
    }

    for (j = 0; j <= KUWAHARA_RADIUS; ++j)  {
        for (i = 0; i <= KUWAHARA_RADIUS; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(st + vec2(i,j) * pixel);
            m2 += c;
            s2 += c * c;
        }
    }

    for (j = 0; j <= KUWAHARA_RADIUS; ++j)  {
        for (i = -KUWAHARA_RADIUS; i <= 0; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(st + vec2(i,j) * pixel);
            m3 += c;
            s3 += c * c;
        }
    }

    float min_sigma2 = 1e+2;
    m0 /= n;
    s0 = abs(s0 / n - m0 * m0);

    float sigma2 = s0.r + s0.g + s0.b;
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

#else

KUWAHARA_TYPE kuwahara(in sampler2D tex, in vec2 st, in vec2 pixel, in int radius) {

    float n = float((KUWAHARA_RADIUS + 1) * (KUWAHARA_RADIUS + 1));
    int i; int j;
    KUWAHARA_TYPE m0 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m1 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m2 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE m3 = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE s0 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s1 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s2 = KUWAHARA_TYPE(0.0); KUWAHARA_TYPE s3 = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE rta = KUWAHARA_TYPE(0.0);
    KUWAHARA_TYPE c;
    
    for (j = -KUWAHARA_RADIUS; j <= 0; ++j)  {
        for (i = -KUWAHARA_RADIUS; i <= 0; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(st + vec2(i,j) * pixel);
            m0 += c;
            s0 += c * c;
        }
    }
    
    for (j = -KUWAHARA_RADIUS; j <= 0; ++j)  {
        for (i = 0; i <= KUWAHARA_RADIUS; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(st + vec2(i,j) * pixel);
            m1 += c;
            s1 += c * c;
        }
    }
    
    for (j = 0; j <= KUWAHARA_RADIUS; ++j)  {
        for (i = 0; i <= KUWAHARA_RADIUS; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(st + vec2(i,j) * pixel);
            m2 += c;
            s2 += c * c;
        }
    }
    
    for (j = 0; j <= KUWAHARA_RADIUS; ++j)  {
        for (i = -KUWAHARA_RADIUS; i <= 0; ++i)  {
            c = KUWAHARA_SAMPLER_FNC(st + vec2(i,j) * pixel);
            m3 += c;
            s3 += c * c;
        }
    }
    
    
    float min_sigma2 = 1e+2;
    m0 /= n;
    s0 = abs(s0 / n - m0 * m0);
    
    float sigma2 = s0.r + s0.g + s0.b;
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

#endif

#endif