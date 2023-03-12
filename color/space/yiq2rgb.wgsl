const yiq2rgb_mat : mat3x3<f32>  = mat3x3<f32>( vec3<f32>(1.0, 0.956, 0.621), vec3<f32>(1.0, -0.272, -0.647), vec3<f32>(1.0, -1.105, 1.702) );

fn yiqToRgb(yiq : vec3<f32>) -> vec3<f32> {
    return yiq2rgb_mat * yiq;
}