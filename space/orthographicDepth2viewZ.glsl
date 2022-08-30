/*
author: 
description: from https://github.com/mrdoob/three.js/blob/master/src/renderers/shaders/ShaderChunk/packing.glsl.js
use: 
license: 
*/

#ifndef FNC_ORTHOGRAPHICDEPTH2VIEWZ
#define FNC_ORTHOGRAPHICDEPTH2VIEWZ
float orthographicDepth2viewZ( const in float linearClipZ, const in float near, const in float far ) {
    return linearClipZ * ( near - far ) - near;
}
#endif