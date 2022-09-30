#ifndef FNC_LIGHT_FALLOFF
#define FNC_LIGHT_FALLOFF
float falloff(float _dist, float _lightRadius) {
    float att = saturate(1.0 - _dist * _dist / (_lightRadius * _lightRadius));
    att *= att;
    return att;
}
#endif