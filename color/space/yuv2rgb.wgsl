
// #ifdef YUV_SDTV
// const yuv2rgb_mat = mat3x3<f32>(
//     vec3<f32>(1.0,       1.0,      1.0),
//     vec3<f32>(0.0,      -0.39465,  2.03211),
//     vec3<f32>(1.13983,  -0.58060,  0.0)
// );
// #else
const yuv2rgb_mat = mat3x3<f32>(
    vec3<f32>(1.0,       1.0,      1.0),
    vec3<f32>(0.0,      -0.21482,  2.12798),
    vec3<f32>(1.28033,  -0.38059,  0.0)
);
// #endif

fn yuv2rgb(yuv: vec3<f32>) -> vec3<f32> {
    return yuv2rgb_mat * yuv;
}

