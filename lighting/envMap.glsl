
#include "../math/powFast.glsl"
#include "../color/tonemap.glsl"

#include "fakeCube.glsl"
#include "toShininess.glsl"

/*
author: Patricio Gonzalez Vivo
description: get enviroment map light comming from a normal direction and acording to some roughness/metallic value. If there is no SCENE_CUBEMAP texture it creates a fake cube
use: <vec3> envMap(<vec3> _normal, <float> _roughness [, <float> _metallic])
options:
    - SCENE_CUBEMAP: pointing to the cubemap texture
    - ENVMAP_MAX_MIP_LEVEL: defualt 8

license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef ENVMAP_MAX_MIP_LEVEL
#define ENVMAP_MAX_MIP_LEVEL 3.0
#endif

#ifndef FNC_ENVMAP
#define FNC_ENVMAP
vec3 envMap(vec3 _normal, float _roughness, float _metallic) {
#if defined(SCENE_CUBEMAP)
    float lod = ENVMAP_MAX_MIP_LEVEL * _roughness;
    return textureCube( SCENE_CUBEMAP, _normal, lod).rgb;
#else
    return fakeCube(_normal, toShininess(_roughness, _metallic));
#endif
}

vec3 envMap(vec3 _normal, float _roughness) {
    return envMap(_normal, _roughness, 1.0);
}
#endif