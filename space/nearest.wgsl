fn nearest(v: vec2f, res: vec2f) -> vec2f {
    let offset = 0.5 / (res - 1.0);
    return floor(v * res) / res + offset;
}