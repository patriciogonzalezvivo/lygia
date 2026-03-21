/*
contributors:  Shadi El Hajj
description: Create a view matrix from camera position and camera rotation (euler angles)
use: <mat4> eulerView(in <vec3> position, in <vec3> euler)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#include "../math/rotate3dX.wgsl"
#include "../math/rotate3dY.wgsl"
#include "../math/rotate3dZ.wgsl"
#include "translate.wgsl"

 fn eulerView(position: vec3f, euler: vec3f) -> mat4x4<f32> {
    let rotZ = rotate3dZ(euler.z);
    let rotX = rotate3dX(euler.x);
    let rotY = rotate3dY(euler.y);
    let identity = mat3x3<f32>(1.0);
    let rotation = rotY * rotX * rotZ * identity;
    return translate(rotation, position);
}
