// https://github.com/glslify/glsl-specular-gaussian
fn specularGaussian(NoH: f32, roughness: f32) -> f32 {
    let theta = acos(NoH);
    let w = theta / roughness;
    return exp(-w*w);
}

fn specularGaussiana(shadingData: ShadingData) -> f32 {
    return specularGaussian(shadingData.NoH, shadingData.roughness);
}
