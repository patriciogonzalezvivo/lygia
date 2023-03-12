fn vibrance(color : vec3<f32>, v:f32) -> vec3<f32> {
    let average = (color.r + color.g + color.b) * 0.333333;
    let mx = max(color.r, max(color.g, color.b));
    let amt = (mx - average) * (-v * 3.0);
    return mix(color, vec3<f32>(mx), amt );
}