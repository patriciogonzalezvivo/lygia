#include "../math/const.wgsl"

fn fisheye2xyz(uv: vec2f) -> vec3f {
    let ndc = uv * 2.0 - 1.0;
    let R = sqrt(ndc.x * ndc.x + ndc.y * ndc.y);
    var dir = vec3(ndc.x / R, 0.0, ndc.y / R);
    let Phi = (R) * PI * 0.52;
    dir.y   = cos(Phi);
    let s = sqrt(1.0 - dir.y * dir.y);
    dir.x *= s;
    dir.z *= s;
    return dir;
}