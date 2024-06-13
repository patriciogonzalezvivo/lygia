#include "lengthSq.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Returns the lenght of a quaternion'
use: <QUAT> quatLength(<QUAT> q)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quatLength(q: vec4f) -> f32 { return sqrt(quatLengthSq(q)); }