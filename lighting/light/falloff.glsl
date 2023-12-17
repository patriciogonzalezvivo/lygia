#include "../../math/saturate.glsl"

#ifndef FNC_LIGHT_FALLOFF
#define FNC_LIGHT_FALLOFF
float falloff(const in float _dist, const in float _lightRadius) {
    float att = saturate(1.0 - _dist * _dist / (_lightRadius * _lightRadius));
    att *= att;
    return att;
}
#endif