fn rgb2luma(color : vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}