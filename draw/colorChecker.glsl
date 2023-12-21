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
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_colorChecker.frag
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

vec4 colorCheckerMacbethXYZ(vec2 uv) {
    vec3 xyz_values[24];
    xyz_values[0] = DARK_SKIN_XYZ;
    xyz_values[1] = LIGHT_SKIN_XYZ;
    xyz_values[2] = BLUE_SKY_XYZ;
    xyz_values[3] = FOLIAGE_XYZ;
    xyz_values[4] = BLUE_FLOWER_XYZ;
    xyz_values[5] = BLUISH_GREEN_XYZ;
    xyz_values[6] = ORANGE_XYZ;
    xyz_values[7] = PURPLISH_BLUE_XYZ;
    xyz_values[8] = MODERATE_RED_XYZ;
    xyz_values[9] = PURPLE_XYZ;
    xyz_values[10] = YELLOW_GREEN_XYZ;
    xyz_values[11] = ORANGE_YELLOW_XYZ;
    xyz_values[12] = BLUE_XYZ;
    xyz_values[13] = GREEN_XYZ;
    xyz_values[14] = RED_XYZ;
    xyz_values[15] = YELLOW_XYZ;
    xyz_values[16] = MAGENTA_XYZ;
    xyz_values[17] = CYAN_XYZ;
    xyz_values[18] = WHITE_XYZ;
    xyz_values[19] = NEUTRAL_80_XYZ;
    xyz_values[20] = NEUTRAL_65_XYZ;
    xyz_values[21] = NEUTRAL_50_XYZ;
    xyz_values[22] = NEUTRAL_35_XYZ;
    xyz_values[23] = BLACK_XYZ;

    vec2 st = vec2(uv.x, 1.0-uv.y);
    st = scale(st, vec2(1.0,1.5));
    st *= vec2(6.0, 4.0);
    vec2 st_i = floor(st);
    vec2 st_f = fract(st);

    vec3 xyz = vec3(0.0);
    int index = 6 * int(st_i.y) + int(st_i.x);
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 64; i++)
        if (i == index) xyz = xyz_values[i];
    #else
    xyz = xyz_values[index];
    #endif
    xyz *= rect(st_f, 0.8);
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(xyz * alpha, alpha);
}

vec4 colorCheckerMacbethLAB(vec2 uv) {
    vec3 lab_values[24];
    lab_values[0] = DARK_SKIN_LAB;
    lab_values[1] = LIGHT_SKIN_LAB;
    lab_values[2] = BLUE_SKY_LAB;
    lab_values[3] = FOLIAGE_LAB;
    lab_values[4] = BLUE_FLOWER_LAB;
    lab_values[5] = BLUISH_GREEN_LAB;
    lab_values[6] = ORANGE_LAB;
    lab_values[7] = PURPLISH_BLUE_LAB;
    lab_values[8] = MODERATE_RED_LAB;
    lab_values[9] = PURPLE_LAB;
    lab_values[10] = YELLOW_GREEN_LAB;
    lab_values[11] = ORANGE_YELLOW_LAB;
    lab_values[12] = BLUE_LAB;
    lab_values[13] = GREEN_LAB;
    lab_values[14] = RED_LAB;
    lab_values[15] = YELLOW_LAB;
    lab_values[16] = MAGENTA_LAB;
    lab_values[17] = CYAN_LAB;
    lab_values[18] = WHITE_LAB;
    lab_values[19] = NEUTRAL_80_LAB;
    lab_values[20] = NEUTRAL_65_LAB;
    lab_values[21] = NEUTRAL_50_LAB;
    lab_values[22] = NEUTRAL_35_LAB;
    lab_values[23] = BLACK_LAB;

    vec2 st = vec2(uv.x, 1.0-uv.y);
    st = scale(st, vec2(1.0,1.5));
    st *= vec2(6.0, 4.0);
    vec2 st_i = floor(st);
    vec2 st_f = fract(st);

    vec3 lab = vec3(0.0);
    int index = 6 * int(st_i.y) + int(st_i.x);
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 64; i++)
        if (i == index) lab = lab_values[i];
    #else
    lab = lab_values[index];
    #endif
    lab *= rect(st_f, 0.8);
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(lab * alpha, alpha);
}

vec4 colorCheckerMacbethLCH(vec2 uv) {
    vec3 lch_values[24];
    lch_values[0] = DARK_SKIN_LCH;
    lch_values[1] = LIGHT_SKIN_LCH;
    lch_values[2] = BLUE_SKY_LCH;
    lch_values[3] = FOLIAGE_LCH;
    lch_values[4] = BLUE_FLOWER_LCH;
    lch_values[5] = BLUISH_GREEN_LCH;
    lch_values[6] = ORANGE_LCH;
    lch_values[7] = PURPLISH_BLUE_LCH;
    lch_values[8] = MODERATE_RED_LCH;
    lch_values[9] = PURPLE_LCH;
    lch_values[10] = YELLOW_GREEN_LCH;
    lch_values[11] = ORANGE_YELLOW_LCH;
    lch_values[12] = BLUE_LCH;
    lch_values[13] = GREEN_LCH;
    lch_values[14] = RED_LCH;
    lch_values[15] = YELLOW_LCH;
    lch_values[16] = MAGENTA_LCH;
    lch_values[17] = CYAN_LCH;
    lch_values[18] = WHITE_LCH;
    lch_values[19] = NEUTRAL_80_LCH;
    lch_values[20] = NEUTRAL_65_LCH;
    lch_values[21] = NEUTRAL_50_LCH;
    lch_values[22] = NEUTRAL_35_LCH;
    lch_values[23] = BLACK_LCH;

    vec2 st = vec2(uv.x, 1.0-uv.y);
    st = scale(st, vec2(1.0,1.5));
    st *= vec2(6.0, 4.0);
    vec2 st_i = floor(st);
    vec2 st_f = fract(st);

    vec3 lch = vec3(0.0);
    int index = 6 * int(st_i.y) + int(st_i.x);
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 64; i++)
        if (i == index) lch = lch_values[i];
    #else
    lch = lch_values[index];
    #endif
    lch *= rect(st_f, 0.8);
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(lch * alpha, alpha);
}

vec4 colorCheckerMacbethXYY(vec2 uv) {
    vec3 xyY_values[24];
    xyY_values[0] = DARK_SKIN_XYY;
    xyY_values[1] = LIGHT_SKIN_XYY;
    xyY_values[2] = BLUE_SKY_XYY;
    xyY_values[3] = FOLIAGE_XYY;
    xyY_values[4] = BLUE_FLOWER_XYY;
    xyY_values[5] = BLUISH_GREEN_XYY;
    xyY_values[6] = ORANGE_XYY;
    xyY_values[7] = PURPLISH_BLUE_XYY;
    xyY_values[8] = MODERATE_RED_XYY;
    xyY_values[9] = PURPLE_XYY;
    xyY_values[10] = YELLOW_GREEN_XYY;
    xyY_values[11] = ORANGE_YELLOW_XYY;
    xyY_values[12] = BLUE_XYY;
    xyY_values[13] = GREEN_XYY;
    xyY_values[14] = RED_XYY;
    xyY_values[15] = YELLOW_XYY;
    xyY_values[16] = MAGENTA_XYY;
    xyY_values[17] = CYAN_XYY;
    xyY_values[18] = WHITE_XYY;
    xyY_values[19] = NEUTRAL_80_XYY;
    xyY_values[20] = NEUTRAL_65_XYY;
    xyY_values[21] = NEUTRAL_50_XYY;
    xyY_values[22] = NEUTRAL_35_XYY;
    xyY_values[23] = BLACK_XYY;

    vec2 st = vec2(uv.x, 1.0-uv.y);
    st = scale(st, vec2(1.0,1.5));
    st *= vec2(6.0, 4.0);
    vec2 st_i = floor(st);
    vec2 st_f = fract(st);

    vec3 xyY = vec3(0.0);
    int index = 6 * int(st_i.y) + int(st_i.x);
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 64; i++)
        if (i == index) xyY = xyY_values[i];
    #else
    xyY = xyY_values[index];
    #endif
    xyY *= rect(st_f, 0.8);
    float alpha = saturate(rect(uv, vec2(1.03,0.69)));
    return vec4(xyY, alpha);
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

vec4 colorCheckerSpyderALAB(vec2 uv) {
    vec3 colors[24];
    colors[0] = LOW_SAT_RED_LAB;
    colors[1] = LOW_SAT_YELLOW_LAB;
    colors[2] = LOW_SAT_GREEN_LAB;
    colors[3] = LOW_SAT_CYAN_LAB;
    colors[4] = LOW_SAT_BLUE_LAB;
    colors[5] = LOW_SAT_MAGENTA_LAB;

    colors[6] = RED_TINT_10_LAB;
    colors[7] = GREEN_TINT_10_LAB;
    colors[8] = BLUE_TINT_10_LAB;
    colors[9] = RED_TONE_90_LAB;
    colors[10] = GREEN_TONE_90_LAB;
    colors[11] = BLUE_TONE_90_LAB;
    
    colors[12] = LIGHTEST_SKIN_LAB;
    colors[13] = LIGHTER_SKIN_LAB;
    colors[14] = MODERATE_SKIN_LAB;
    colors[15] = MEDIUM_SKIN_LAB;
    colors[16] = DEEP_SKIN_LAB;
    colors[17] = GRAY_95_LAB;

    colors[18] = GRAY_05_LAB;
    colors[19] = GRAY_10_LAB;
    colors[20] = GRAY_30_LAB;
    colors[21] = GRAY_50_LAB;
    colors[22] = GRAY_70_LAB;
    colors[23] = GRAY_90_LAB;

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
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(color * alpha, alpha);
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

vec4 colorCheckerSpyderBLAB(vec2 uv) {
    vec3 colors[24];
    colors[0] = AQUA_LAB;
    colors[1] = LAVANDER_LAB;
    colors[2] = EVERGREEN_LAB;
    colors[3] = STEEL_BLUE_LAB;
    colors[4] = CLASSIC_LIGHT_SKIN_LAB;
    colors[5] = CLASSIC_DARK_SKIN_LAB;

    colors[6] = PRIMARY_ORANGE_LAB;
    colors[7] = BLUEPRINT_LAB;
    colors[8] = PINK_LAB;
    colors[9] = VIOLET_LAB;
    colors[10] = APPLE_GREEN_LAB;
    colors[11] = SUNFLOWER_LAB;
    
    colors[12] = PRIMARY_CYAN_LAB;
    colors[13] = PRIMARY_MAGENTA_LAB;
    colors[14] = PRIMARY_YELLOW_LAB;
    colors[15] = PRIMARY_RED_LAB;
    colors[16] = PRIMARY_GREEN_LAB;
    colors[17] = PRIMARY_BLUE_LAB;

    colors[18] = CARD_WHITE_LAB;
    colors[19] = GRAY_20_LAB;
    colors[20] = GRAY_40_LAB;
    colors[21] = GRAY_60_LAB;
    colors[22] = GRAY_80_LAB;
    colors[23] = BLACK_LAB;

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
    float alpha = rect(uv, vec2(1.03,0.69));
    return vec4(color * alpha, alpha);
}

vec4 colorCheckerSpyder(vec2 uv) {
    return colorCheckerSpyderB(uv);
}

vec4 colorChecker (vec2 uv){
    return COLORCHECKER_FNC(uv);
}

#endif