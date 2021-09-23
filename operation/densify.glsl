/*
author: Patricio Gonzalez Vivo
description: simple densification
use: densifyBox(<sampler2D> texture, <vec2> st, <vec2> pixels_scale, <int> passes)
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_DENSIFY
#define FNC_DENSIFY

vec3 densifyBox(sampler2D tex, vec2 st, vec2 pixel, int passes) {
    vec3 color = texture2D(tex, st).rgb;

    if (color == vec3(0.0)) {
        float weight = 0.0;

        int kernelSize = 3;
        for (int k = 0; k < passes; k++) {
            float f_kernelSize = float(kernelSize);
            for (int j = 0; j < kernelSize; j++) {
                float y = -.5 * (f_kernelSize - 1.) + float(j);
                for (int i = 0; i < kernelSize; i++) {
                float x = -.5 * (f_kernelSize - 1.) + float(i);
                    vec3 value = texture2D(tex, st + vec2(x, y) * pixel).rgb;
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

vec3 densifyGaussian(sampler2D tex, vec2 st, vec2 pixel, int passes) {
    vec3 color = texture2D(tex, st).rgb;

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

                    vec3 value = texture2D(tex, st + xy * pixel).rgb;
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

vec3 densify(sampler2D tex, vec2 st, vec2 pixel, int passes) {
    return densifyGaussian(tex, st, pixel, passes);
}

#endif