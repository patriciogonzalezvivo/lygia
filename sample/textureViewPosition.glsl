#include "../space/depth2viewZ.glsl"
#include "../space/screen2viewPosition.glsl"

/*
author: Patricio Gonzalez Vivo
description: sampler the view Positiong from depthmap texture 
use: <vec4> textureViewPosition(<sampler2D> texDepth, <vec2> st [, <float> near, <float> far])
options:
    - SAMPLE_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - CAMERA_NEAR_CLIP: required
    - CAMERA_FAR_CLIP: required
    - CAMERA_ORTHOGRAPHIC_PROJECTION, if it's not present is consider a PERECPECTIVE camera
    - CAMERA_PROJECTION_MATRIX: mat4 matrix with camera projection
    - INVERSE_CAMERA_PROJECTION_MATRIX: mat4 matrix with the inverse camara projection

license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef SAMPLE_FNC
#define SAMPLE_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef FNC_VIEWPOSITION
#define FNC_VIEWPOSITION

vec4 textureViewPosition(sampler2D texDepth, const in vec2 st, const in float near, const in float far) {
    float depth = SAMPLE_FNC(texDepth, st).r;
    float viewZ = depth2viewZ(depth, near, far);
    return screen2viewPosition(st, depth, viewZ);
}

#if defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)
float textureViewPosition(sampler2D texDepth, const in vec2 st) {
    return textureViewPosition( texDepth, st, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP); 
}
#endif

#endif