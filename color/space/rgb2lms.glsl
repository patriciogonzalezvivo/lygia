/*
author: Patricio Gonzalez Vivo  
description: convert rgb to LMS. LMS (long, medium, short), is a color space which represents the response of the three types of cones of the human eye, named for their responsivity (sensitivity) peaks at long, medium, and short wavelengths. https://en.wikipedia.org/wiki/LMS_color_space
use: <vec3|vec4> rgb2lms(<vec3|vec4> rgb)
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_RGB2LMS
#define FNC_RGB2LMS
vec3 rgb2lms(vec3 rgb) {
    return vec3(
        (17.8824 * rgb.r) + (43.5161 * rgb.g) + (4.11935 * rgb.b),
        (3.45565 * rgb.r) + (27.1554 * rgb.g) + (3.86714 * rgb.b),
        (0.0299566 * rgb.r) + (0.184309 * rgb.g) + (1.46709 * rgb.b)
    );
}
vec4 rgb2lms(vec4 rgb) { vec4(rgb.rgb, rgb.a); }
#endif