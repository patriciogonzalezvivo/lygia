fn YCbCr2rgb(ycbcr: vec3f) -> vec3f {
    let cb = ycbcr.y - 0.5;
    let cr = ycbcr.z - 0.5;
    let y = ycbcr.x;
    let r = 1.402 * cr;
    let g = -0.344 * cb - 0.714 * cr;
    let b = 1.772 * cb;
    return vec3f(r, g, b) + y;
}
