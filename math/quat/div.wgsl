/*
contributors: Patricio Gonzalez Vivo
description: "Quaternion division \n"
use: <QUAT> quatDiv(<QUAT> a, <QUAT> b)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quatDiv(q: vec4f, s: f32) -> vec4f { return vec4f(q.xyz / s, q.w / s); }
