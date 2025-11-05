import lygia::math::consts::PI;

/*
contributors: Patricio Gonzalez Vivo
description: 3D vector to equirect 2D projection
use: <vec2> xyz2equirect(<vec2> d)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn xyz2equirect(d: vec3f) -> vec2f {
    return vec2f(atan2(d.z, d.x) + PI, acos(-d.y)) / vec2f(2.0 * PI, PI);
}
