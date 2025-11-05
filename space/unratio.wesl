/*
contributors: Patricio Gonzalez Vivo
description: unFix the aspect ratio
use: ratio(vec2 st, vec2 st_size)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn unratio(st: vec2f, s: vec2f) -> vec2f {
    return vec2f(st.x, st.y * (s.x / s.y) + (s.y * 0.5 - s.x * 0.5) / s.y);
}
