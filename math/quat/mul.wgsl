/*
contributors: Patricio Gonzalez Vivo
description: 'Quaternion multiplication. Based on http://mathworld.wolfram.com/Quaternion.html'
use: <vec4f> quatMul(<vec4f> a, <vec4f> b)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quatMul(q1: vec4f, q2: vec4f) {
    return vec4f(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}