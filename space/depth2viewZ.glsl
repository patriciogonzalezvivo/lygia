/*
original_author: Patricio Gonzalez Vivo 
description: Convert camera depth to view depth. based on https://github.com/mrdoob/three.js/blob/master/src/renderers/shaders/ShaderChunk/packing.glsl.js
use: <float> depth2viewZ( <float> depth [, <float> near, <float> far] ) 
options: 
    - CAMERA_NEAR_CLIP
    - CAMERA_FAR_CLIP
    - CAMERA_ORTHOGRAPHIC_PROJECTION, if it's not present is consider a PERECPECTIVE camera
license: |
    Copyright (c) 2022 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_DEPTH2VIEWZ
#define FNC_DEPTH2VIEWZ
float depth2viewZ( const in float depth, const in float near, const in float far ) {
    #if defined(CAMERA_ORTHOGRAPHIC_PROJECTION) 
    // ORTHOGRAPHIC
    return depth * ( near - far ) - near;
    #else 
    // PERSPECCTIVE
    return ( near * far ) / ( ( far - near ) * depth - far );
    #endif
}

#if defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)
float depth2viewZ( const in float depth) {
    return depth2viewZ( depth, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP); 
}
#endif

#endif