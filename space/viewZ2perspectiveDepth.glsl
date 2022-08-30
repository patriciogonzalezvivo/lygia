/*
author: 
description: from https://github.com/mrdoob/three.js/blob/master/src/renderers/shaders/ShaderChunk/packing.glsl.js
use: 
license: 
*/

#ifndef FNC_VIEWZ2ORTHOGRAPHICDEPTH
#define FNC_VIEWZ2ORTHOGRAPHICDEPTH
float viewZ2orthographicDepth( const in float viewZ, const in float near, const in float far ) {
    return ( viewZ + near ) / ( near - far );
}
#endif