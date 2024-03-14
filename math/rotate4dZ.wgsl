/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
*/

fn rotate4dZ(r: f32) -> mat4x4<f32> {
    return mat4x4<f32>( vec4f(cos(r),-sin(r),0.,0),
                        vec4f(sin(r),cos(r),0.,0.),
                        vec4f(0.,0.,1.,0.),
                        vec4f(0.,0.,0.,1.));
}
