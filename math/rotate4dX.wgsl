/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
*/

fn rotate4dX(r: f32) -> mat4x4<f32> {
    return mat4x4<f32> (vec4f(1.,0.,0.,0),
                        vec4f(0.,cos(r),-sin(r),0.),
                        vec4f(0.,sin(r),cos(r),0.),
                        vec4f(0.,0.,0.,1.));
}
