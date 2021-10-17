#include "../math/powFast.glsl"
#include "material/shininess.glsl"

#ifndef FNC_FAKECUBE
#define FNC_FAKECUBE

vec3 fakeCube(vec3 _normal, float _shininnes) {
    vec3 rAbs = abs(_normal);
    return vec3( powFast(max(max(rAbs.x, rAbs.y), rAbs.z) + 0.005, _shininnes) );
}

vec3 fakeCube(vec3 _normal) {
    return fakeCube(_normal, materialShininess() );
}

#endif