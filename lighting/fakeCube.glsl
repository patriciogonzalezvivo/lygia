#include "../math/powFast.glsl"
#include "material/shininess.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: creates a fake cube and returns the value giving a normal direction
use: <vec3> fakeCube(<vec3> _normal [, <float> _shininnes])
*/

#ifndef FNC_FAKECUBE
#define FNC_FAKECUBE

vec3 fakeCube(const in vec3 _normal, const in float _shininnes) {
    vec3 rAbs = abs(_normal);
    return vec3( powFast(max(max(rAbs.x, rAbs.y), rAbs.z) + 0.005, _shininnes) );
}

vec3 fakeCube(const in vec3 _normal) {
    return fakeCube(_normal, materialShininess() );
}

#endif