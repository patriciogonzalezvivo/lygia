#include "../../math/const.glsl"

#ifndef FNC_ASHIKHMIN
#define FNC_ASHIKHMIN
float ashikhmin(const in float NoH, const in float roughness) {
    // Ashikhmin 2007, "Distribution-based BRDFs"
    float a2 = roughness * roughness;
    float cos2h = NoH * NoH;
    float sin2h = max(1.0 - cos2h, 0.0078125); // 2^(-14/2), so sin2h^2 > 0 in fp16
    float sin4h = sin2h * sin2h;
    float cot2 = -cos2h / (a2 * sin2h);
    return 1.0 / (PI * (4.0 * a2 + 1.0) * sin4h) * (4.0 * exp(cot2) + sin4h);
}
#endif