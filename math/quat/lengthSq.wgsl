/*
contributors: Patricio Gonzalez Vivo
description: 'Returns the squared length of a quaternion.'
use: <QUAT> quatLengthSq(<QUAT> q)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quatLengthSq(q: vec4f) -> f32 { return dot(q.xyz, q.xyz) + q.w * q.w; }