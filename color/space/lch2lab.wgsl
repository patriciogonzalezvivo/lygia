/*
contributors: Patricio Gonzalez Vivo
description: "Converts a Lch ot Lab color space. \nNote: LCh is simply Lab but converted to polar coordinates (in degrees).\n"
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn lch2lab(lch: vec3f) -> vec3f {
    return vec3f(
        lch.x,
        lch.y * cos(lch.z * 0.01745329251),
        lch.y * sin(lch.z * 0.01745329251)
    );
}