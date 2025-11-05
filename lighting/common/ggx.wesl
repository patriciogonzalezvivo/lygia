import lygia::math::consts::{INV_PI, PI};
import lygia::math::saturateMediump::saturateMediump;

// Walter et al. 2007, "Microfacet Models for Refraction through Rough Surfaces"

// This one is not great on mediump. Read next comment
fn GGX(NoH: f32, roughness: f32) -> f32 {
    let oneMinusNoHSquared = 1.0 - NoH * NoH;
    let a = NoH * roughness;
    let k = roughness / (oneMinusNoHSquared + a * a);
    let d = k * k * INV_PI;
    return saturateMediump(d);
}

// Precision version with platform-specific optimization
// Matches GLSL: float GGX(const in vec3 N, const in vec3 H, const in float NoH, float roughness)
fn GGXPrecise(N: vec3f, H: vec3f, NoH: f32, roughness: f32) -> f32 {
    @if(TARGET_MOBILE) {
        // In mediump, there are two problems computing 1.0 - NoH^2
        // 1) 1.0 - NoH^2 suffers floating point cancellation when NoH^2 is close to 1 (highlights)
        // 2) NoH doesn't have enough precision around 1.0
        // Both problem can be fixed by computing 1-NoH^2 in highp and providing NoH in highp as well

        // However, we can do better using Lagrange's identity:
        //      ||a x b||^2 = ||a||^2 ||b||^2 - (a . b)^2
        // since N and H are unit vectors: ||N x H||^2 = 1.0 - NoH^2
        // This computes 1.0 - NoH^2 directly (which is close to zero in the highlights and has
        // enough precision).
        // Overall this yields better performance, keeping all computations in mediump
        let NxH = cross(N, H);
        let oneMinusNoHSquared = dot(NxH, NxH);

        let a = NoH * roughness;
        let k = roughness / (oneMinusNoHSquared + a * a);
        let d = k * k * INV_PI;
        return saturateMediump(d);
    } @else {
        let oneMinusNoHSquared = 1.0 - NoH * NoH;

        let a = NoH * roughness;
        let k = roughness / (oneMinusNoHSquared + a * a);
        let d = k * k * INV_PI;
        return saturateMediump(d);
    }
}

fn importanceSamplingGGX(u: vec2f, roughness: f32) -> vec3f {
    let a2 = roughness * roughness;
    let phi = 2.0 * PI * u.x;
    let cosTheta2 = (1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y);
    let cosTheta = sqrt(cosTheta2);
    let sinTheta = sqrt(1.0 - cosTheta2);
    return vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
