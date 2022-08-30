/*
author: Patricio Gonzalez Vivo
description: derive view surface position from screen coordinates and depth 
use: <vec4> gooch(<vec4> baseColor, <vec3> normal, <vec3> light, <vec3> view, <float> roughness)
options:
    - GOOCH_WARM: defualt vec3(0.25, 0.15, 0.0)
    - GOOCH_COLD: defualt vec3(0.0, 0.0, 0.2)
    - GOOCH_SPECULAR: defualt vec3(1.0, 1.0, 1.0)
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_COORD:       in GlslViewer is  v_lightCoord
    - LIGHT_SHADOWMAP:   in GlslViewer is u_lightShadowMap
    - LIGHT_SHADOWMAP_SIZE: in GlslViewer is 1024.0

license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef PROJECTION_MATRIX
#if defined(GLSLVIEWER)
#define PROJECTION_MATRIX u_projectionMatrix
#else
#define PROJECTION_MATRIX projectionMatrix
#endif
#endif

#ifndef INVERSE_PROJECTION_MATRIX
#if defined(GLSLVIEWER)
#include "lygia/math/inverse.glsl"
#define INVERSE_PROJECTION_MATRIX inverse(u_projectionMatrix)
#else
#define INVERSE_PROJECTION_MATRIX inverseProjectionMatrix
#endif
#endif

#ifndef FNC_VIEWPOSITION
#define FNC_VIEWPOSITION

// source: https://github.com/mrdoob/three.js/blob/dev/examples/js/shaders/SSAOShader.js
vec3 viewPosition(const in vec2 screenPosition, const in float depth) {
    float clipW = PROJECTION_MATRIX[2][3] * depth + PROJECTION_MATRIX[3][3];
    vec4 clipPosition = vec4((vec3(screenPosition, depth) - 0.5) * 2.0, 1.0);
    clipPosition *= clipW;
    return (INVERSE_PROJECTION_MATRIX * clipPosition).xyz;
}

vec3 viewPosition( const in vec2 screenPosition, const in float depth, const in float viewZ ) {
    float clipW = PROJECTION_MATRIX[2][3] * viewZ + PROJECTION_MATRIX[3][3];
    vec4 clipPosition = vec4( ( vec3(screenPosition, depth ) - 0.5 ) * 2.0, 1.0 );
    clipPosition *= clipW; // unprojection.
    return (INVERSE_PROJECTION_MATRIX * clipPosition ).xyz;
}

#endif