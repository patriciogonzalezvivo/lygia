/*
contributors: Patricio Gonzalez Vivo
description: given a quaternion, returns a rotation 3x3 matrix
use: <mat3> quat2mat3(<QUAT> Q)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quat2mat3(q: vec4f) -> mat3x3<f32> {
    let qxx = q.x * q.x;
    let qyy = q.y * q.y;
    let qzz = q.z * q.z;
    let qxz = q.x * q.z;
    let qxy = q.x * q.y;
    let qyw = q.y * q.w;
    let qzw = q.z * q.w;
    let qyz = q.y * q.z;
    let qxw = q.x * q.w;

    return mat3x3<f32>(
        vec3f(1.0 - 2.0 * (qyy + qzz), 2.0 * (qxy - qzw), 2.0 * (qxz + qyw)),
        vec3f(2.0 * (qxy + qzw), 1.0 - 2.0 * (qxx + qzz), 2.0 * (qyz - qxw)),
        vec3f(2.0 * (qxz - qyw), 2.0 * (qyz + qxw), 1.0 - 2.0 * (qxx + qyy))
    );
}