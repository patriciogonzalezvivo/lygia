/*
original_author: Patricio Gonzalez Vivo
description: |
    Quaternion division 
use: <QUAT> quatDiv(<QUAT> a, <QUAT> b) 
*/

fn quatDiv(q: vec4f, s: f32) -> vec4f { return vec4f(q.xyz / s, q.w / s); }
