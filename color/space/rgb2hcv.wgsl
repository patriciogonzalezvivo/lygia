fn rgb2hcv(rgb: vec3f) -> vec3f {
    let P = (rgb.g < rgb.b) ? vec4f(rgb.bg, -1.0, 2.0/3.0) : vec4f(rgb.gb, 0.0, -1.0/3.0);
    let Q = (rgb.r < P.x) ? vec4f(P.xyw, rgb.r) : vec4f(rgb.r, P.yzx);
    let C = Q.x - min(Q.w, Q.y);
    let H = abs((Q.w - Q.y) / (6.0 * C + 1e-10) + Q.z);
    return vec3f(H, C, Q.x);
}