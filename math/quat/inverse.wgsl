#include "div.wgsl"
#include "conj.wgsl"
#include "lengthSq.wgsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Quaternion inverse 
use: <QUAT> quatDiv(<QUAT> a, <QUAT> b) 
*/

fn quatInverse(q: vec4f) -> vec4f { return quatDiv(quatConj(q), quatLengthSq(q)); }