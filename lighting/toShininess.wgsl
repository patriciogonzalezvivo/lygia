fn toShininess(roughness: f32, metallic: f32) -> f32 {
    var s = 0.95 - roughness * 0.5;
    s *= s;
    s *= s;
    return s * (80.0 + 160.0 * (1.0-metallic));
}