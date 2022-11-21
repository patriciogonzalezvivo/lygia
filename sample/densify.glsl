#include "../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: simple densification
use: densifyBox(<sampler2D> texture, <vec2> st, <vec2> pixels_scale, <int> passes)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
*/

#ifndef FNC_SAMPLEDENSIFY
#define FNC_SAMPLEDENSIFY

vec3 sampleDensifyBox(sampler2D tex, vec2 st, vec2 pixel, int passes) {
    vec3 color = SAMPLER_FNC(tex, st).rgb;

    if (color == vec3(0.0)) {
        float weight = 0.0;

        int kernelSize = 3;
        for (int k = 0; k < passes; k++) {
            float f_kernelSize = float(kernelSize);
            for (int j = 0; j < kernelSize; j++) {
                float y = -.5 * (f_kernelSize - 1.) + float(j);
                for (int i = 0; i < kernelSize; i++) {
                float x = -.5 * (f_kernelSize - 1.) + float(i);
                    vec3 value = SAMPLER_FNC(tex, st + vec2(x, y) * pixel).rgb;
                    if (value != vec3(0.0)) {
                        color += value;
                        weight++;
                    }
                }
            }

            kernelSize += 2;
        }

        color /= weight;
    }

    return color;
}

vec3 sampleDensifyGaussian(sampler2D tex, vec2 st, vec2 pixel, int passes) {
    vec3 color = SAMPLER_FNC(tex, st).rgb;

    if (dot(color,color) == 0.0) {
        int kernelSize = 3;
        float accumWeight = 1.;
        const float k = .39894228;
        float kernelSize2 = float(kernelSize) * float(kernelSize);

        for (int k = 0; k < passes; k++) {
            float f_kernelSize = float(kernelSize);

            for (int j = 0; j < kernelSize; j++) {
                float y = -.5 * (f_kernelSize - 1.) + float(j);

                for (int i = 0; i < kernelSize; i++) {
                    float x = -.5 * (f_kernelSize - 1.) + float(i);
                    vec2 xy = vec2(x, y);

                    vec3 value = SAMPLER_FNC(tex, st + xy * pixel).rgb;
                    if (dot(value,value) > 0.0) {
                        float weight = (k / f_kernelSize * exp(-(x * x + y * y) / (2. * kernelSize2)));
                        color += weight * value;
                        accumWeight += weight;
                    }
                }
            }
            kernelSize += 2;
        }

        color /= accumWeight;
    }

    return color;
}

vec3 sampleDensify(sampler2D tex, vec2 st, vec2 pixel, int passes) {
    return sampleDensifyGaussian(tex, st, pixel, passes);
}

#endif