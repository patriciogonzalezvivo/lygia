#include "linear2gamma.wgsl"

vec3 xyz2rgb(in vec3 c) {
    const M = mat3x3<f32>(  vec3<f32>(3.2404542, -0.9692660,  0.0556434),
                            vec3<f32>(-1.5371585,  1.8760108, -0.2040259),
                            vec3<f32>(-0.4985314,  0.0415560,  1.0572252));
    let v = M * (c / 100.0);
    let c0 = (1.055 * linear2gamma(v)) - 0.055;
    let c1 = 12.92 * v;
    return mix(c0, c1, step(v, vec3<f32>(0.0031308)));
}
