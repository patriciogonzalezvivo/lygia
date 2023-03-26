fn saturationMatrix(amount : f32) -> mat3x3<f32> {
    let lum = vec3<f32>(0.3086, 0.6094, 0.0820 );
    let invAmount = 1.0 - amount;
    return mat3x3<f32>( vec3<f32>(lum.x * invAmount) + vec3<f32>(amount, .0, .0), 
                        vec3<f32>(lum.y * invAmount) + vec3<f32>( .0, amount, .0),
                        vec3<f32>(lum.z * invAmount) + vec3<f32>( .0, .0, amount));
}