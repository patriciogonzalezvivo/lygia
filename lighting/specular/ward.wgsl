// https://github.com/glslify/glsl-specular-ward
fn specularWard(L: vec3f, N: vec3f, V: vec3f, fiber: vec3f, shinyParallel: f32, shinyPerpendicular: f32) -> f32 {
    let NdotL = dot(N, L);
    let NdotR = dot(N, V);
    let fiberParallel = normalize(fiber);
    let fiberPerpendicular = normalize(cross(N, fiber));

    if (NdotL < 0.0 || NdotR < 0.0)
        return 0.0;

    let H = normalize(L + V);

    let NdotH = dot(N, H);
    let XdotH = dot(fiberParallel, H);
    let YdotH = dot(fiberPerpendicular, H);

    let coeff = sqrt(NdotL/NdotR) / (12.5663706144 * shinyParallel * shinyPerpendicular);
    let theta = (pow(XdotH/shinyParallel, 2.0) + pow(YdotH/shinyPerpendicular, 2.0)) / (1.0 + NdotH);

    return coeff * exp(-2.0 * theta);
}
