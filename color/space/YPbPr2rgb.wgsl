// #ifdef YPBPR_SDTV
// const YPbPr2rgb_mat = mat3x3<f32>( 
//     vec3<f32>(1.0,     1.0,      1.0),
//     vec3<f32>(0.0,    -0.344,    1.772),
//     vec3<f32>(1.402,  -0.714,    0.0)
// );
// #else
const YPbPr2rgb_mat = mat3x3<f32>( 
    vec3<f32>(1.0,     1.0,      1.0),
    vec3<f32>(0.0,    -0.187,    1.856),
    vec3<f32>(1.575,  -0.468,    0.0)
);
// #endif

fn YPbPr2rgb(rgb: vec3<f32>) -> vec3<f32> {
    return YPbPr2rgb_mat * rgb;
}
