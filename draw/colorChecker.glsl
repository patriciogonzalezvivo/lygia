#include "../space/scale.glsl"
#include "../color/checker.glsl"
#include "../sdf/crossSDF.glsl"
#include "rect.glsl"
#include "stroke.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: draw a color checker (Macbeth or Spyder)
use: 
    - colorChecker(<vec2> uv)
    - colorCheckerMacbeth(<vec2> uv)
    - colorCheckerSpyder(<vec2> uv)
options:
    - COLORCHECKER_FNC: function to use to draw the color checker
*/

#ifndef COLORCHECKER_FNC
#define COLORCHECKER_FNC(UV) colorCheckerMacbeth(UV)
#endif

#ifndef COLOR_CHECKER
#define COLOR_CHECKER

vec4 colorCheckerMacbeth (vec2 uv) {
    vec3 colors[24];
    colors[0] = DARK_SKIN;
    colors[1] = LIGHT_SKIN;
    colors[2] = BLUE_SKY;
    colors[3] = FOLIAGE;
    colors[4] = BLUE_FLOWER;
    colors[5] = BLUISH_GREEN;
    colors[6] = ORANGE;
    colors[7] = PURPLISH_BLUE;
    colors[8] = MODERATE_RED;
    colors[9] = PURPLE;
    colors[10] = YELLOW_GREEN;
    colors[11] = ORANGE_YELLOW;
    colors[12] = BLUE;
    colors[13] = GREEN;
    colors[14] = RED;
    colors[15] = YELLOW;
    colors[16] = MAGENTA;
    colors[17] = CYAN;
    colors[18] = WHITE;
    colors[19] = NEUTRAL_80;
    colors[20] = NEUTRAL_65;
    colors[21] = NEUTRAL_50;
    colors[22] = NEUTRAL_35;
    colors[23] = BLACK;

    vec2 st = vec2(uv.x, 1.0-uv.y);
    st = scale(st, vec2(1.0,1.5));
    st *= vec2(6.0, 4.0);
    vec2 st_i = floor(st);
    vec2 st_f = fract(st);

    vec3 color = vec3(0.0);
    int index = 6 * int(st_i.y) + int(st_i.x);
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 64; i++)
        if (i == index) color = colors[i];
    #else
    color = colors[index];
    #endif
    color *= rect(st_f, 0.8);
    color += fill(crossSDF(uv, 2.), .015);
    color += saturate(
                stroke(rectSDF(uv, vec2(1.015, 0.68)), 1., 0.01) -
                rect(uv, vec2(.966, 1.)) - rect(uv, vec2(1.1, .63))
            );
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(saturate(color) * alpha, alpha);
}

vec4 colorCheckerSpyderA(vec2 uv) {
    vec3 colors[24];
    colors[0] = LOW_SAT_RED;
    colors[1] = LOW_SAT_YELLOW;
    colors[2] = LOW_SAT_GREEN;
    colors[3] = LOW_SAT_CYAN;
    colors[4] = LOW_SAT_BLUE;
    colors[5] = LOW_SAT_MAGENTA;

    colors[6] = RED_TINT_10;
    colors[7] = GREEN_TINT_10;
    colors[8] = BLUE_TINT_10;
    colors[9] = RED_TONE_90;
    colors[10] = GREEN_TONE_90;
    colors[11] = BLUE_TONE_90;
    
    colors[12] = LIGHTEST_SKIN;
    colors[13] = LIGHTER_SKIN;
    colors[14] = MODERATE_SKIN;
    colors[15] = MEDIUM_SKIN;
    colors[16] = DEEP_SKIN;
    colors[17] = GRAY_95;

    colors[18] = GRAY_05;
    colors[19] = GRAY_10;
    colors[20] = GRAY_30;
    colors[21] = GRAY_50;
    colors[22] = GRAY_70;
    colors[23] = GRAY_90;

    vec2 st = uv;
    st = scale(st, vec2(1.0,1.5));
    st *= vec2(6.0, 4.0);
    vec2 st_i = floor(st);
    vec2 st_f = fract(st);

    vec3 color = vec3(0.0);
    int index = 6 * int(st_i.y) + int(st_i.x);
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 64; i++)
        if (i == index) color = colors[i];
    #else
    color = colors[index];
    #endif
    color *= rect(st_f, 0.8);
    color += fill(crossSDF(uv, 2.), .015);
    color += saturate(
                stroke(rectSDF(uv, vec2(1.015, 0.68)), 1., 0.01) -
                rect(uv, vec2(.966, 1.)) - rect(uv, vec2(1.1, .63))
            );
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(saturate(color) * alpha, alpha);
}

vec4 colorCheckerSpyderB(vec2 uv) {
    vec3 colors[24];
    colors[0] = AQUA;
    colors[1] = LAVANDER;
    colors[2] = EVERGREEN;
    colors[3] = STEEL_BLUE;
    colors[4] = CLASSIC_LIGHT_SKIN;
    colors[5] = CLASSIC_DARK_SKIN;

    colors[6] = PRIMARY_ORANGE;
    colors[7] = BLUEPRINT;
    colors[8] = PINK;
    colors[9] = VIOLET;
    colors[10] = APPLE_GREEN;
    colors[11] = SUNFLOWER;
    
    colors[12] = PRIMARY_CYAN;
    colors[13] = PRIMARY_MAGENTA;
    colors[14] = PRIMARY_YELLOW;
    colors[15] = PRIMARY_RED;
    colors[16] = PRIMARY_GREEN;
    colors[17] = PRIMARY_BLUE;

    colors[18] = CARD_WHITE;
    colors[19] = GRAY_20;
    colors[20] = GRAY_40;
    colors[21] = GRAY_60;
    colors[22] = GRAY_80;
    colors[23] = BLACK;

    vec2 st = vec2(uv.x, 1.0-uv.y);
    st = scale(st, vec2(1.0,1.5));
    st *= vec2(6.0, 4.0);
    vec2 st_i = floor(st);
    vec2 st_f = fract(st);

    vec3 color = vec3(0.0);
    int index = 6 * int(st_i.y) + int(st_i.x);
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 64; i++)
        if (i == index) color = colors[i];
    #else
    color = colors[index];
    #endif
    color *= rect(st_f, 0.8);
    color += fill(crossSDF(uv, 2.), .015);
    color += saturate(
                stroke(rectSDF(uv, vec2(1.015, 0.68)), 1., 0.01) -
                rect(uv, vec2(.966, 1.)) - rect(uv, vec2(1.1, .63))
            );
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(saturate(color) * alpha, alpha);
}

vec4 colorCheckerSpyder(vec2 uv) {
    return colorCheckerSpyderB(uv);
}

vec4 colorChecker (vec2 uv){
    return COLORCHECKER_FNC(uv);
}

#endif