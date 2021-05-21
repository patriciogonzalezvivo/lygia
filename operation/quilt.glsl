/*
author: Patricio Gonzalez Vivo
description: convertes QUILT of tiles into something the LookingGlass Volumetric display can render
use: quilt(<sampler2D> texture, <vec4> calibration, <vec3> tile, <vec2> st, <vec2> resolution)
options:
    QUILT_FLIPSUBP: 
    QUILT_SAMPLE_FNC(POS_UV): Function used to sample into the normal map texture, defaults to texture2D(tex,POS_UV)
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef QUILT_SAMPLE_FNC
#define QUILT_SAMPLE_FNC(UV) texture2D(tex, UV)
#endif

#ifndef FNC_QUILT
#define FNC_QUILT
vec2 quiltMap(vec3 tile, vec2 pos, float a) {
    vec2 tile2 = tile.xy - 1.0;
    vec2 dir = vec2(-1.0);

    a = fract(a) * tile.y;
    tile2.y += dir.y * floor(a);
    a = fract(a) * tile.x;
    tile2.x += dir.x * floor(a);
    return (tile2 + pos) / tile.xy;
}

vec3 quilt(sampler2D tex, vec4 calibration, vec3 tile, vec2 st, vec2 resolution) {
    const float pitch = -resolution.x / calibration.x  * calibration.y * sin(atan(abs(calibration.z)));
    const float tilt = resolution.y / (resolution.x * calibration.z);
    const float subp = 1.0 / (3.0 * resolution.x);
    const float subp2 = subp * pitch;

    float a = (-st.x - st.y * tilt) * pitch - calibration.w;

    vec3 color = vec3(0.0);
    #ifdef QUILT_FLIPSUBP
    color.r = QUILT_SAMPLE_FNC( quiltMap(u_holoPlayTile, st, a-2.0*subp2) ).r;
    color.g = QUILT_SAMPLE_FNC( quiltMap(u_holoPlayTile, st, a-subp2) ).g;
    color.b = QUILT_SAMPLE_FNC( quiltMap(u_holoPlayTile, st, a) ).b;
    #else
    color.r = QUILT_SAMPLE_FNC( quiltMap(u_holoPlayTile, st, a) ).r;
    color.g = QUILT_SAMPLE_FNC( quiltMap(u_holoPlayTile, st, a-subp2) ).g;
    color.b = QUILT_SAMPLE_FNC( quiltMap(u_holoPlayTile, st, a-2.0*subp2) ).b;
    #endif
    return color;
}
#endif