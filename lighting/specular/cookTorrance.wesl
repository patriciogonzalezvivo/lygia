import package::lighting::common::ggx::GGX;

fn specularCookTorrance(L: vec3f, N: vec3f, V: vec3f, NoV: f32, NoL: f32, roughness: f32, fresnel: f32) -> f32 {
    // Half angle vector
    let H = normalize(L + V);

    // Geometric term
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.000001);

    let x = 2.0 * NoH / VoH;
    let G = min(1.0, min(x * NoV, x * NoL));
    
    // Distribution term
    let D = GGX(N, H, NoH, roughness);

    // Fresnel term
    let F = pow(1.0 - NoV, fresnel);

    // Multiply terms and done
    return max(G * F * D / max(PI * NoV * NoL, 0.00001), 0.0);
}
