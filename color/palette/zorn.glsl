#include "pigments.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    [Zorn palette](https://www.jacksonsart.com/blog/2021/02/02/colour-mixing-exploring-the-zorn-palette/) where:
    - 0: Titanium White
    - 1: Yellow Ocher
    - 2: Cadmium Red
    - 3: Ivory Black
*/

#ifndef ZORN_TOTAL
#define ZORN_TOTAL 3
#endif

#ifndef FNC_ZORN
#define FNC_ZORN

vec3 zorn(const int index) {
    vec3 color[4];
    color[0] = TITANIUM_WHITE
    color[1] = YELLOW_OCHER;
    color[2] = CADMIUM_RED;
    color[3] = IVORY_BLACK;

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < ZORN_TOTAL; i++)
        if (i == index) return colors[i];
    #else
    return colors[index];
    #endif
}

#endif