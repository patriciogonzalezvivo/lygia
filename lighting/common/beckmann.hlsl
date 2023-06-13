#ifndef FNC_BECKMANN
#define FNC_BECKMANN
float beckmann(float _NoH, float roughness) {
    float NoH = max(_NoH, 0.0001);
    float cos2Alpha = NoH * NoH;
    float tan2Alpha = (cos2Alpha - 1.0) / cos2Alpha;
    float roughness2 = roughness * roughness;
    float denom = 3.141592653589793 * roughness2 * cos2Alpha * cos2Alpha;
    return exp(tan2Alpha / roughness2) / denom;
}
#endif