/*
Author: [Johan Ismael, Patricio Gonzalez Vivo]
description: Use LUT textures to modify colors (vec4 and vec3) or a position in a gradient (vec2 and floats)
use: lut(<sampler2D> texture, <vec4|vec3|vec2|float> value [, int row])
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo & Johan Ismael.
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
  #ifndef LUT_INVERT
  texc.y = 1. - texc.y; 
  #endif

  // sample the 2 adjacent blue slices
  vec3 b0 = texture2D(tex_lut, texc).xyz;
  vec3 b1 = texture2D(tex_lut, vec2(texc.x + LUT_SIZE.w, texc.y)).xyz;

  // blend between the 2 adjacent blue slices
  color.xyz = mix(b0, b1, bFrac);

  return color;
}

vec4 lut(in sampler2D tex_lut, in vec4 color) {
  return lut(tex_lut, color, 0);
}

vec3 lut(in sampler2D tex_lut, in vec3 color, in int offset) {
  return lut(tex_lut, vec4(color, 1.), offset).rgb;
}

vec3 lut(in sampler2D tex_lut, in vec3 color) {
  return lut(tex_lut, color, 0).rgb;
}

#endif