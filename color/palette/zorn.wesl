/*
contributors: Patricio Gonzalez Vivo
description: |
    Zorn palette (https://www.jacksonsart.com/blog/2021/02/02/colour-mixing-exploring-the-zorn-palette/) where:
    - 0: Titanium White
    - 1: Yellow Ochre
    - 2: Cadmium Red
    - 3: Ivory Black
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_zorn.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

import lygia::color::palette::pigments::{TITANIUM_WHITE, YELLOW_OCHRE, CADMIUM_RED, IVORY_BLACK};

const ZORN_TOTAL: u32 = 4u;

fn zorn(index: i32) -> vec3f {
    var colors = array<vec3f, ZORN_TOTAL>(
        TITANIUM_WHITE,
        YELLOW_OCHRE,
        CADMIUM_RED,
        IVORY_BLACK
    );
    let idx = index % i32(ZORN_TOTAL);
    return colors[idx];
}
