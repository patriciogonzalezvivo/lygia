import lygia::math::consts::{PI, TWO_PI};

/*
contributors: Patricio Gonzalez Vivo
description: equirect 2D projection to 3D vector
use: <vec3> equirect2xyz(<vec2> uv)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn equirect2xyz(uv: vec2f) -> vec3f {
    let Phi = PI - uv.y * PI;
    let Theta = uv.x * TWO_PI;
    var dir = vec3f(cos(Theta), 0.0, sin(Theta));
    dir.y = cos(Phi);
    dir.x *= sqrt(1.0 - dir.y * dir.y);
    dir.z *= sqrt(1.0 - dir.y * dir.y);
    return dir;
}
