fn gaussian(      d: f32, sigma: f32) -> f32 { 
    return exp(-(d*d) / (2.0 * sigma * sigma)); 
}

fn gaussian2(d: vec2<f32>, sigma: f32) -> f32 { 
    return exp(-(d.x*d.x + d.y*d.y) / (2.0 * sigma * sigma)); 
}

fn gaussian3(d: vec3<f32>, sigma: f32) -> f32 { 
    return exp(-(d.x*d.x + d.y*d.y + d.z*d.z) / (2.0 * sigma * sigma)); 
}

fn gaussian4(d: vec4<f32>, sigma: f32) -> f32 { 
    return exp(-(d.x*d.x + d.y*d.y + d.z*d.z + d.w*d.w) / (2.0 * sigma * sigma)); 
}