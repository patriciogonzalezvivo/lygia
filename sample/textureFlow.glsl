/*
author: Patricio Gonzalez Vivp
description: sample a texture with a looping flow animation, using a direction to push, an elapse time and a cycle.
use: textureFlow(<sampler2D> tex, <vec2> st, <vec2> dir, <float> time, <float> cycle)
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_TEXTUREFLOW
#define FNC_TEXTUREFLOW
vec4 textureFlow(sampler2D tex, vec2 st, vec2 dir, float time, float cycle) {
    float halfCycle = cycle * 0.5;

    float flowOffset0 = mod(time, cycle);
    float flowOffset1 = mod(time + halfCycle, cycle);

    float phase0 = flowOffset0;
    float phase1 = flowOffset1;

    // Sample normal map.
    vec4 A = texture2D(tex, (st + dir * phase0) );
    vec4 B = texture2D(tex, (st + dir * phase1) );

    float f = (abs(halfCycle - flowOffset0) / halfCycle);
    return mix( A, B, f );
}
#endif