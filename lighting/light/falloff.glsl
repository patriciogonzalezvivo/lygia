#ifndef FNC_LIGHT_FALLOFF
#define FNC_LIGHT_FALLOFF
float falloff(float _dist, float _lightRadius) {
    float att = clamp(1.0 - _dist * _dist / (_lightRadius * _lightRadius), 0.0, 1.0);
    att *= att;
    return att;
}
#endif