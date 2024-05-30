/*
original_author: Patricio Gonzalez Vivo
description: |
    Returns the squared length of a quaternion.
    
use: <QUAT> quatLengthSq(<QUAT> q) 
*/

fn quatLengthSq(q: vec4f) -> f32 { return dot(q.xyz, q.xyz) + q.w * q.w; }