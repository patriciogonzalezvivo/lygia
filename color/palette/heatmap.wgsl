fn heatmap(v: f32) -> vec3f{
    let r = v * 2.1 - vec3f(1.8, 1.14, 0.3);
    return 1.0 - r * r;
}