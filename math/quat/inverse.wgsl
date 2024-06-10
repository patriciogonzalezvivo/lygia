#include "div.wgsl"
#include "conj.wgsl"
#include "lengthSq.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: "Quaternion inverse \n"
use: <QUAT> quatDiv(<QUAT> a, <QUAT> b)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quatInverse(q: vec4f) -> vec4f { return quatDiv(quatConj(q), quatLengthSq(q)); }