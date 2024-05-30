/*
original_author: Patricio Gonzalez Vivo
description: |
    Quaternion addition 
use: <QUAT> quatAdd(<QUAT> a, <QUAT> b) 
*/

fn quatAdd(a: vec4f, b: vec4f) -> vec4f { return vec4f(a.xyz + b.xyz, a.w + b.w); }
