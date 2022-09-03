/*
author: Patricio Gonzalez Vivo
description: get screen coordinates from view position 
use: <vec2> view2screenPosition(<vec3> viewPosition )
options:
    - PROJECTION_MATRIX: mat4 matrix with camera projection
    - RESOLUTION_SCREEN: vec2 with screen resolution

license: |
    Copyright (c) 2022 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef PROJECTION_MATRIX
#define PROJECTION_MATRIX u_projectionMatrix
#endif

#ifndef FNC_VIEW2SCREENPOSITION
#define FNC_VIEW2SCREENPOSITION
vec2 view2screenPosition(vec3 viewPosition){
    vec4 clip = PROJECTION_MATRIX * vec4(viewPosition, 1.0);
    vec2 xy = clip.xy;
    xy /= clip.w;
    xy = (xy + 1.0) * 0.5;
    return xy;
}
#endif