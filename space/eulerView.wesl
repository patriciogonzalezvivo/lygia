import lygia::math::rotate3dX::rotate3dX;
import lygia::math::rotate3dY::rotate3dY;
import lygia::math::rotate3dZ::rotate3dZ;
import lygia::space::translate::translate;

/*
contributors:  Shadi El Hajj
description: Create a view matrix from camera position and camera rotation (euler angles)
use: <mat4> eulerView(in <vec3> position, in <vec3> euler)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn eulerView(position: vec3f, euler: vec3f) -> mat4x4f {
    let rotZ = rotate3dZ(euler.z);
    let rotX = rotate3dX(euler.x);
    let rotY = rotate3dY(euler.y);
    let identity = mat3x3f(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    );
    let rotation = rotY * rotX * rotZ * identity;
    return translate(rotation, position);
}