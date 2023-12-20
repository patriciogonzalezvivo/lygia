// #ifdef YUV_SDTV
// const RGB2YUV = mat3x3<f32>(
//     vec3f(0.299, -0.14713,  0.615),
//     vec3f(0.587, -0.28886, -0.51499),
//     vec3f(0.114,  0.436,   -0.10001)
// );
// #else
const RGB2YUV = mat3x3<f32>(
    vec3f(0.2126,  -.09991, .615),
    vec3f(0.7152,  -.33609,-.55861),
    vec3f(0.0722,   .426,  -.05639)
);
// #endif

fn rgb2yuv(rgb: vec3f) -> vec3f {
    return RGB2YUV * rgb;
}
