/*
author: Patricio Gonzalez Vivo
description: Converts a RGB normal map into normal vectors
use: normalMap(<sampler2D> texture, <vec2> st, <vec2> pixel)
options:
  NORMALMAP_Z: Steepness of z before normalization, defaults to .01
  NORMALMAP_SAMPLER_FNC(POS_UV): Function used to sample into the normal map texture, defaults to texture2D(tex,POS_UV).r
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef NORMALMAP_Z
#define NORMALMAP_Z .01
#endif

#ifndef NORMALMAP_SAMPLER_FNC
#define NORMALMAP_SAMPLER_FNC(POS_UV) texture2D(tex,POS_UV).r
#endif

#ifndef FNC_NORMALMAP
#define FNC_NORMALMAP
vec3 normalMap(sampler2D tex, vec2 st, vec2 pixel) {
    float center = NORMALMAP_SAMPLER_FNC(st);

    float topLeft    = NORMALMAP_SAMPLER_FNC(st - pixel);

    float left = NORMALMAP_SAMPLER_FNC(st - vec2(pixel.x, .0));
    float bottomLeft = NORMALMAP_SAMPLER_FNC(st + vec2(-pixel.x, pixel.y));
    float top    = NORMALMAP_SAMPLER_FNC(st - vec2(.0, pixel.y));
    float bottom = NORMALMAP_SAMPLER_FNC(st + vec2(.0, pixel.y));
    float topRight = NORMALMAP_SAMPLER_FNC(st + vec2(pixel.x, -pixel.y));
    float right    = NORMALMAP_SAMPLER_FNC(st + vec2(pixel.x, .0));
    float bottomRight = NORMALMAP_SAMPLER_FNC(st + pixel);
    
    float dX = topRight + 2. * right + bottomRight - topLeft - 2. * left - bottomLeft;
    float dY = bottomLeft + 2. * bottom + bottomRight - topLeft - 2. * top - topRight;

    return normalize(vec3(dX, dY, NORMALMAP_Z) );
}
#endif