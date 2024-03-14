#include "lengthSq.wgsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Returns the lenght of a quaternion
    
use: <QUAT> quatLength(<QUAT> q) 
*/

fn quatLength(q: vec4f) -> f32 { return sqrt(quatLengthSq(q)); }