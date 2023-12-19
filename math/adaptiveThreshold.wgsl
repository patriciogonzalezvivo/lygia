fn adaptiveThreshold(v: f32, blur_v: f32, b: f32) -> f32 {
    return step(blur_v + b, v);
}
