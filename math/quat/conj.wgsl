/*
contributors: Patricio Gonzalez Vivo
description: given a quaternion, returns its conjugate
use: <QUAT> quatConj(<QUAT> Q)
*/

fn quatConj(q: vec4f) -> vec4f { return vec4f(-q.xyz, q.w); }