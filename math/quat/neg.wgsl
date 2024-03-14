/*
original_author: Patricio Gonzalez Vivo
description: |
    Quaternion negative. 
use: <QUAT> quatNeg(<QUAT> a) 
*/

fn quatNeg(q: vec4f) -> vec4f { return vec4f(-q.xyz, -q.w); }