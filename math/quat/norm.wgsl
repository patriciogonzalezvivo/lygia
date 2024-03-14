#include "length.wgsl"
#include "div.wgsl"

/*
original_author: Patricio Gonzalez Vivo
description: returns a normalized quaternion
use: <QUAT> quatNorm(<QUAT> Q)
*/

fn quatNorm(q: vec4f) -> vec4f { return quatDiv(q, quatLength(q)); }
