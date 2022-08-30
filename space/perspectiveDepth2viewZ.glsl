/*
author: 
description: from https://github.com/mrdoob/three.js/blob/master/src/renderers/shaders/ShaderChunk/packing.glsl.js
use: 
license: 
*/

#ifndef FNC_PERSPECTIVEDEPTH2VIEWZ
#define FNC_PERSPECTIVEDEPTH2VIEWZ
float perspectiveDepth2viewZ( const in float invClipZ, const in float near, const in float far ) {
    return ( near * far ) / ( ( far - near ) * invClipZ - far );
}
#endif