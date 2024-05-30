#include "2mat3.wgsl"
#include "../toMat4.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: given a quaternion, returns a rotation 4x4 matrix
use: <mat4> quat2mat4(<QUAT> Q)
*/

fn quat2mat4(v q: vec4f) -> mat4x4<f32> { return toMat4(quat2mat3(q)); }
