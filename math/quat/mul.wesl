/*
contributors: Patricio Gonzalez Vivo
description: 'Quaternion multiplication. Based on http://mathworld.wolfram.com/Quaternion.html'
use: quatMul(q1: vec4f, q2: vec4f) -> vec4f | quatMulScalar(q: vec4f, s: f32) -> vec4f | quatMulVec3(q: vec4f, v: vec3f) -> vec3f
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quatMul(q1: vec4f, q2: vec4f) -> vec4f {
    return vec4f(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}

fn quatMulScalar(q: vec4f, s: f32) -> vec4f { return vec4f(q.xyz * s, q.w * s); }

// Rotate a vec3 by a quaternion
// Based on: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
fn quatMulVec3(q: vec4f, v: vec3f) -> vec3f {
    let qv = cross(q.xyz, v);
    let qv2 = cross(q.xyz, qv + q.w * v);
    return v + 2.0 * qv2;
}