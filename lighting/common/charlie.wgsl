#include "../../math/const.wgsl"

fn charlie(NoH: f32, roughness: f32) -> f32 {
    // Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF"
    let invAlpha = 1.0 / roughness;
    let cos2h = NoH * NoH;
    float sin2h = max(1.0 - cos2h, 0.0078125); // 2^(-14/2), so sin2h^2 > 0 in fp16
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / TAU;
}
