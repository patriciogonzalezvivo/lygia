/*
contributors: Patricio Gonzalez Vivo
description: |
    Quaternion multiplication. Based on http://mathworld.wolfram.com/Quaternion.html
use: <vec4f> quatMul(<vec4f> a, <vec4f> b) 
*/

fn quatMul(q1: vec4f, q2: vec4f) {
    return vec4f(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}