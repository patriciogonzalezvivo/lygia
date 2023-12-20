fn xyY2xyz(xyY: vec3f) -> vec3f {
    let Y = xyY.z;
    let f = 1.0/xyY.y;
    let x = Y * xyY.x * f;
    let z = Y * (1.0 - xyY.x - xyY.y) * f;
    return vecf(x, Y, z);
}
