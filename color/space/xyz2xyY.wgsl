fn xyz2xyY(xyz: vec3f) -> vec3f {
    let Y = xyz.y;
    let f = 1.0 / (xyz.x + xyz.y + xyz.z);
    let x = xyz.x * f;
    let y = xyz.y * f;
    return vec3f(x, y, Y);
}