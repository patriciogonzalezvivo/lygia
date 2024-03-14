/*
original_author: Patricio Gonzalez Vivo
description: |
    Quaternion substraction. 
use: <QUAT> quatNeg(<QUAT> a, <QUAT> b) 
*/

fn quatSub(a: vec4f, b: vec4f) -> vec4f { return vec4f(a.xyz - b.xyz, a.w - b.w); }
