/*
contributors: Inigo Quiles
description: generate the SDF of a cylinder

*/

// vertical
fn cylinderSDF(p: vec3f, h: vec2f) -> f32 {
    let d = abs(vec2f(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,vec2f(0.0)));
}

fn cylinderSDF1(p: vec3f, h: f32) -> f32 {
    return cylinderSDF(p, vec2f(h));
}

fn cylinderSDF2(p: vec3f, h: f32, r: f32) -> f32 {
    let d = abs(vec2f(length(p.xz),p.y)) - vec2f(h,r);
    return min(max(d.x,d.y),0.0) + length(max(d,vec2f(0.0)));
}

// arbitrary orientation
fn cylinderSDF4(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let baba = dot(ba,ba);
    let paba = dot(pa,ba);

    let x = length(pa*baba-ba*paba) - r*baba;
    let y = abs(paba-baba*0.5)-baba*0.5;
    let x2 = x*x;
    let y2 = y*y*baba;
    let d = select(-min(x2,y2), select(0.0, x2, x>0.0)+select(0.0, y2, y>0.0), max(x,y)<0.0);
    return sign(d)*sqrt(abs(d))/baba;
}