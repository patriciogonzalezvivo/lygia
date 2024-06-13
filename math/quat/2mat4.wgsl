#include "2mat3.wgsl"
#include "../toMat4.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: given a quaternion, returns a rotation 4x4 matrix
use: <mat4> quat2mat4(<QUAT> Q)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quat2mat4(v q: vec4f) -> mat4x4<f32> { return toMat4(quat2mat3(q)); }
