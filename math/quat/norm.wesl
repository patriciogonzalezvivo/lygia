import lygia::math::quat::length::{quatLength};
import lygia::math::quat::div::{quatDiv};

/*
contributors: Patricio Gonzalez Vivo
description: returns a normalized quaternion
use: quatNorm(q: vec4f) -> vec4f
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quatNorm(q: vec4f) -> vec4f { return quatDiv(q, quatLength(q)); }