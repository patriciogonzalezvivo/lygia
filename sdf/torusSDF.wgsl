fn torusSDF(p: vec3f, t: vec2f ) -> f32 { 
    return length( vec2f(length(p.xz) - t.x, p.y) ) - t.y;
}