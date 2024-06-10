#include "../math/powFast.hlsl"
#include "material/shininess.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Creates a fake cube and returns the value giving a normal direction
use: <float3> fakeCube(<float3> _normal [, <float> _shininnes])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
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