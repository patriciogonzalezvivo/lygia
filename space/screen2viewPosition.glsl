/*
author: Patricio Gonzalez Vivo
description: derive view surface position from screen coordinates and depth 
use: <vec3> screen2viewPosition( const in <vec2> screenPosition, const in <float> depth, const in <float> viewZ )
options:
    - PROJECTION_MATRIX: mat4 matrix with camera projection
    - INVERSE_PROJECTION_MATRIX: mat4 matrix with the inverse camara projection

license: |
    Copyright (c) 2022 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef PROJECTION_MATRIX
#define PROJECTION_MATRIX u_projectionMatrix
#endif

#ifndef INVERSE_PROJECTION_MATRIX
// #define INVERSE_PROJECTION_MATRIX u_inverseProjectionMatrix
#define INVERSE_PROJECTION_MATRIX inverse(PROJECTION_MATRIX)
#endif

#ifndef FNC_SCREEN2VIEWPOSITION
#define FNC_SCREEN2VIEWPOSITION

vec3 screen2viewPosition( const in vec2 screenPosition, const in float depth, const in float viewZ ) {
    float clipW = PROJECTION_MATRIX[2][3] * viewZ + PROJECTION_MATRIX[3][3];
    vec4 clipPosition = vec4( ( vec3(screenPosition, depth ) - 0.5 ) * 2.0, 1.0 ) * clipW;
    return (INVERSE_PROJECTION_MATRIX * clipPosition ).xyz;
}

#endif