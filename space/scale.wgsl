fn scale(st: vec2f, s: vec2f) -> vec2f {
    return (st - 0.5) * s + 0.5;
}