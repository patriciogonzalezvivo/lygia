const rgb2yiq_mat : mat3x3<f32>  = mat3x3<f32>( vec3<f32>(0.299, 0.587, 0.114), vec3<f32>(0.596, -0.274, -0.322), vec3<f32>(0.212, -0.523, 0.311) );

fn rgb2yiq(rgb : vec3<f32>) -> vec3<f32> {
    return rgb2yiq_mat * rgb;
}
