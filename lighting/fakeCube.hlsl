#include "../math/powFast.hlsl"
#include "material/shininess.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: creates a fake cube and returns the value giving a normal direction
use: <float3> fakeCube(<float3> _normal [, <float> _shininnes])
*/

#ifndef FNC_FAKECUBE
#define FNC_FAKECUBE

float3 fakeCube(float3 _normal, float _shininnes) {
    float3 rAbs = abs(_normal);
    float v = powFast(max(max(rAbs.x, rAbs.y), rAbs.z) + 0.005, _shininnes );
    return float3(v, v, v);
}

float3 fakeCube(float3 _normal) {
    return fakeCube(_normal, materialShininess() );
}

#endif