/*
contributors: Patricio Gonzalez Vivo
description: "Converts a LCh to Lab color space. \nNote: LCh is simply Lab but converted to polar coordinates (in degrees).\n"
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn lab2lch(lab: vec3f) -> vec3f {
    return vec3f(
        lab.x,
        sqrt(dot(lab.yz, lab.yz)),
        atan(lab.z, lab.y) * 57.2957795131
    );
}