#include "../../math/const.wgsl"

/*
contributors:  Shadi El Hajj
description: Light Falloff equation, based on the model in Brian Karis' paper "Real Shading in Unreal Engine 4"
use: float falloff(float dist, float lightRadius)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn falloff(dist: f32, lightRadius: f32) -> f32 {
    let dr = dist/lightRadius;
    let att = saturate(1.0 - dr*dr*dr*dr);
    att *= att;
    return att / (dist * dist + EPSILON);
}
