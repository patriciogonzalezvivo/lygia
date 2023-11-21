fn rectSDF_round(p: vec2f , b: vec2f, r: f32) -> f32 {
    let d = abs(p - 0.5) * 4.2 - b + r;
    return min(max(d.x, d.y), 0.0) + length(vec2f( max(d.x, 0.0), max(d.y, 0.0) )) - r;   
}

fn rectSDF(st: vec2f, s: vec2f) -> f32 {
    let uv = st * 2.0 - 1.0;
    return max( abs(uv.x / s.x),
                abs(uv.y / s.y) );
}