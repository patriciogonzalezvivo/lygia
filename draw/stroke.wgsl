fn stroke(x: f32, size: f32, w: f32, edge: f32) -> f32 {
    return saturate(smoothstep(size - edge, size + edge, x + w * 0.5) - smoothstep(size - edge, size + edge, x - w * 0.5));
}