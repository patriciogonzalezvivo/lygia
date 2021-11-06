#include "../math/saturate.glsl"
/*
Author: [Matt DesLauriers, Johan Ismael, Patricio Gonzalez Vivo]
description: Use LUT textures to modify colors (vec4 and vec3) or a position in a gradient (vec2 and floats)
use: lut(<sampler2D> texture, <vec4|vec3|vec2|float> value [, int row])
license: |
    The MIT License (MIT) Copyright (c) 2014 Matt DesLauriers
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    
    The MIT License (MIT) Copyright (c) 2017 Patricio Gonzalez Vivo & Johan Ismael.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef LUT_N_ROWS
#define LUT_N_ROWS 1
#endif

#ifndef LUT_CELL_SIZE
#define LUT_CELL_SIZE 32.0
#endif

#ifndef FNC_LUT
#define FNC_LUT

#ifdef LUT_SQUARE 
vec4 lut(in sampler2D tex_lut, in vec4 color, in int offset) {
    float blueColor = color.b * 63.0;
    
    vec2 quad1 = vec2(0.0);
    quad1.y = floor(floor(blueColor) / 8.0);
    quad1.x = floor(blueColor) - (quad1.y * 8.0);
    
    vec2 quad2 = vec2(0.0);
    quad2.y = floor(ceil(blueColor) / 8.0);
    quad2.x = ceil(blueColor) - (quad2.y * 8.0);
    
    vec2 texPos1 = vec2(0.0);
    texPos1.x = (quad1.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.r);
    texPos1.y = (quad1.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.g);

    #ifdef LUT_FLIP_Y
    texPos1.y = 1.0-texPos1.y;
    #endif
    
    vec2 texPos2 = vec2(0.0);
    texPos2.x = (quad2.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.r);
    texPos2.y = (quad2.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.g);

    #ifdef LUT_FLIP_Y
    texPos2.y = 1.0-texPos2.y;
    #endif
    
    vec4 b0 = texture2D(tex_lut, texPos1);
    vec4 b1 = texture2D(tex_lut, texPos2);

    return mix(b0, b1, fract(blueColor));
}

#else
// Data about how the LUTs rows are encoded
const float LUT_WIDTH = LUT_CELL_SIZE*LUT_CELL_SIZE;
const float LUT_OFFSET = 1./ float( LUT_N_ROWS );
const vec4 LUT_SIZE = vec4(LUT_WIDTH, LUT_CELL_SIZE, 1./LUT_WIDTH, 1./LUT_CELL_SIZE);

// Apply LUT to a COLOR
// ------------------------------------------------------------
vec4 lut(in sampler2D tex_lut, in vec4 color, in int offset) {
    vec3 scaledColor = clamp(color.rgb, vec3(0.), vec3(1.)) * (LUT_SIZE.y - 1.);
    float bFrac = fract(scaledColor.z);

    // offset by 0.5 pixel and fit within range [0.5, width-0.5]
    // to prevent bilinear filtering with adjacent colors
    vec2 texc = (.5 + scaledColor.xy) * LUT_SIZE.zw;

    // offset by the blue slice
    texc.x += (scaledColor.z - bFrac) * LUT_SIZE.w;
    texc.y *= LUT_OFFSET;
    texc.y += float(offset) * LUT_OFFSET;
    #ifndef LUT_FLIP_Y
    texc.y = 1. - texc.y; 
    #endif

    // sample the 2 adjacent blue slices
    vec4 b0 = texture2D(tex_lut, texc);
    vec4 b1 = texture2D(tex_lut, vec2(texc.x + LUT_SIZE.w, texc.y));

    // blend between the 2 adjacent blue slices
    color = mix(b0, b1, bFrac);

    return color;
}
#endif

vec4 lut(in sampler2D tex_lut, in vec4 color) { return lut(tex_lut, color, 0); }
vec3 lut(in sampler2D tex_lut, in vec3 color, in int offset) { return lut(tex_lut, vec4(color, 1.), offset).rgb; }
vec3 lut(in sampler2D tex_lut, in vec3 color) { return lut(tex_lut, color, 0).rgb; }

#endif