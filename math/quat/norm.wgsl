#include "length.wgsl"
#include "div.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: returns a normalized quaternion
use: <QUAT> quatNorm(<QUAT> Q)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quatNorm(q: vec4f) -> vec4f { return quatDiv(q, quatLength(q)); }
