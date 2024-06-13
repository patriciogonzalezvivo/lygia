/*
contributors: Patricio Gonzalez Vivo
description: "Quaternion negative. \n"
use: <QUAT> quatNeg(<QUAT> a)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quatNeg(q: vec4f) -> vec4f { return vec4f(-q.xyz, -q.w); }