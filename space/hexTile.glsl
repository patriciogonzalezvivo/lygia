#include "scale.glsl"

/*
author: Patricio Gonzalez Vivo
description: make some hexagonal tiles. XY provide coordenates of the tile. While Z provides the distance to the center of the tile
use: <vec3> hexTile(<vec2> st [, <float> scale])
options:
    HEXTILE_SIZE
    HEXTILE_H
    HEXTILE_S
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_HEXTILE
#define FNC_HEXTILE
vec3 hexTile(in vec2 st, in float scl) {
    // this is hack to scale the hexagon to be more squared
    st = scale(st, vec2(1.24,1.)*scl) + vec2(.5,0.);

    vec3 q = vec3(st, .0);
    q.z = -.5 * q.x - q.y;

    float z = -.5 * q.x - q.y;
    q.y -= .5 * q.x;

    vec3 i = floor(q+.5);
    float s = floor(i.x + i.y + i.z);
    vec3 d = abs(i-q);

    // TODO: all this ifs should be avoided
    if( d.x >= d.y && d.x >= d.z ) i.x -= s;
    else if( d.y >= d.x && d.y >= d.z )	i.y -= s;
    else i.z -= s;

    vec2 coord = vec2(i.x, (i.y - i.z + (1.-mod(i.x, 2.)))/2.);
    float dist = length(st - vec2(coord.x, coord.y - .5*mod(i.x-1., 2.)));
    return vec3(coord, dist);
}

vec3 hexTile(in vec2 st) { return hexTile(st, 1.0); }

#endif