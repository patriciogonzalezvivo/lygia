const yiq2rgb_mat : mat3x3<f32>  = mat3x3<f32>( 
    vec3<f32>(1.0,  0.9469,  0.6235), 
    vec3<f32>(1.0, -0.2747, -0.6357), 
    vec3<f32>(1.0, -1.1085,  1.7020) );

fn yiqToRgb(yiq : vec3<f32>) -> vec3<f32> {
    return yiq2rgb_mat * yiq;
}