#include "pigments.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    [Zorn palette](https://www.jacksonsart.com/blog/2021/02/02/colour-mixing-exploring-the-zorn-palette/) where:
    - 0: Titanium White
    - 1: Yellow Ocher
    - 2: Cadmium Red
    - 3: Ivory Black
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_zorn.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef ZORN_TOTAL
#define ZORN_TOTAL 4
#endif

#ifndef FNC_ZORN
#define FNC_ZORN
vec3 zorn(int index) {
    vec3 colors[4];
    colors[0] = TITANIUM_WHITE;
    colors[1] = YELLOW_OCHRE;
    colors[2] = CADMIUM_RED;
    colors[3] = IVORY_BLACK;

    index = int(mod(float(index), float(ZORN_TOTAL)));
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < ZORN_TOTAL; i++)
        if (i == index) return colors[i];
    #else
    return colors[index];
    #endif
}
#endif