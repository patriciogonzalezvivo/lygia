/*
contributors: Patricio Gonzalez Vivo
description: |
    MacBeth ColorChecker values from:
        - http://en.wikipedia.org/wiki/ColorChecker
        - http://kurtmunger.com/color_checkerid277.html
        - http://www.rags-int-inc.com/phototechstuff/macbethtarget/
        - https://babelcolor.com/index_htm_files/RGB%20Coordinates%20of%20the%20Macbeth%20ColorChecker.pdf

    WESL conditionals:
    - @if(CIE_D50): Use D50 illuminant for XYZ/LAB/LCH values (default: D65)
use:
    - <vec3f> macbeth(<i32> index)
    - <vec3f> macbethXYZ(<i32> index)
    - <vec3f> macbethLAB(<i32> index)
    - <vec3f> macbethLCH(<i32> index)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_colorChecker.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const MACBETH_TOTAL: u32 = 24u;

// Row 1: Natural colors (sRGB)
const DARK_SKIN: vec3f = vec3f(0.46017, 0.33059, 0.27477);
const LIGHT_SKIN: vec3f = vec3f(0.769, 0.576, 0.506);
const BLUE_SKY: vec3f = vec3f(0.356, 0.472, 0.609);
const FOLIAGE: vec3f = vec3f(0.357, 0.427, 0.247);
const BLUE_FLOWER: vec3f = vec3f(0.518, 0.506, 0.694);
const BLUISH_GREEN: vec3f = vec3f(0.384, 0.749, 0.675);

// Row 2: Miscellaneous colors (sRGB)
const ORANGE: vec3f = vec3f(0.867, 0.487, 0.184);
const PURPLISH_BLUE: vec3f = vec3f(0.290, 0.357, 0.671);
const MODERATE_RED: vec3f = vec3f(0.769, 0.333, 0.384);
const PURPLE: vec3f = vec3f(0.365, 0.231, 0.420);
const YELLOW_GREEN: vec3f = vec3f(0.624, 0.745, 0.227);
const ORANGE_YELLOW: vec3f = vec3f(0.894, 0.635, 0.160);

// Row 3: Primary and secondary colors (sRGB)
const BLUE: vec3f = vec3f(0.176, 0.247, 0.584);
const GREEN: vec3f = vec3f(0.239, 0.588, 0.290);
const RED: vec3f = vec3f(0.690, 0.224, 0.227);
const YELLOW: vec3f = vec3f(0.925, 0.784, 0.094);
const MAGENTA: vec3f = vec3f(0.749, 0.309, 0.598);
const CYAN: vec3f = vec3f(0.000, 0.537, 0.659);

// Row 4: Grayscale (sRGB)
const WHITE: vec3f = vec3f(0.956, 0.956, 0.945);
const NEUTRAL_80: vec3f = vec3f(0.789, 0.797, 0.797);
const NEUTRAL_65: vec3f = vec3f(0.635, 0.643, 0.643);
const NEUTRAL_50: vec3f = vec3f(0.475, 0.478, 0.478);
const NEUTRAL_35: vec3f = vec3f(0.329, 0.333, 0.337);
const BLACK: vec3f = vec3f(0.200, 0.200, 0.204);

// XYZ color space (D50 vs D65 illuminant)
@if(CIE_D50) const DARK_SKIN_XYZ: vec3f = vec3f(12.354, 10.896, 5.498);
@else const DARK_SKIN_XYZ: vec3f = vec3f(11.684, 10.637, 7.242);
@if(CIE_D50) const LIGHT_SKIN_XYZ: vec3f = vec3f(39.602, 35.073, 19.617);
@else const LIGHT_SKIN_XYZ: vec3f = vec3f(37.29, 34.404, 25.454);
@if(CIE_D50) const BLUE_SKY_XYZ: vec3f = vec3f(17.275, 18.663, 26.177);
@else const BLUE_SKY_XYZ: vec3f = vec3f(17.916, 19.032, 34.703);
@if(CIE_D50) const FOLIAGE_XYZ: vec3f = vec3f(11.284, 13.786, 5.479);
@else const FOLIAGE_XYZ: vec3f = vec3f(10.906, 13.784, 7.155);
@if(CIE_D50) const BLUE_FLOWER_XYZ: vec3f = vec3f(24.969, 23.688, 33.857);
@else const BLUE_FLOWER_XYZ: vec3f = vec3f(25.42, 23.896, 44.94);
@if(CIE_D50) const BLUISH_GREEN_XYZ: vec3f = vec3f(30.838, 42.168, 35.407);
@else const BLUISH_GREEN_XYZ: vec3f = vec3f(31.342, 43.112, 46.048);
@if(CIE_D50) const ORANGE_XYZ: vec3f = vec3f(41.4, 32.052, 5.12);
@else const ORANGE_XYZ: vec3f = vec3f(38.041, 30.531, 6.671);
@if(CIE_D50) const PURPLISH_BLUE_XYZ: vec3f = vec3f(12.767, 11.597, 30.374);
@else const PURPLISH_BLUE_XYZ: vec3f = vec3f(14.025, 12.012, 40.373);
@if(CIE_D50) const MODERATE_RED_XYZ: vec3f = vec3f(30.96, 20.368, 10.532);
@else const MODERATE_RED_XYZ: vec3f = vec3f(28.491, 19.254, 13.978);
@if(CIE_D50) const PURPLE_XYZ: vec3f = vec3f(8.795, 6.668, 10.99);
@else const PURPLE_XYZ: vec3f = vec3f(8.81, 6.597, 14.816);
@if(CIE_D50) const YELLOW_GREEN_XYZ: vec3f = vec3f(35.701, 45.082, 9.377);
@else const YELLOW_GREEN_XYZ: vec3f = vec3f(33.992, 44.969, 11.835);
@if(CIE_D50) const ORANGE_YELLOW_XYZ: vec3f = vec3f(49.37, 44.438, 6.215);
@else const ORANGE_YELLOW_XYZ: vec3f = vec3f(45.74, 42.941, 8.056);
@if(CIE_D50) const BLUE_XYZ: vec3f = vec3f(7.305, 6.022, 22.182);
@else const BLUE_XYZ: vec3f = vec3f(8.374, 6.347, 29.467);
@if(CIE_D50) const GREEN_XYZ: vec3f = vec3f(15.229, 23.542, 8.222);
@else const GREEN_XYZ: vec3f = vec3f(14.881, 23.932, 10.401);
@if(CIE_D50) const RED_XYZ: vec3f = vec3f(22.59, 13.543, 4.08);
@else const RED_XYZ: vec3f = vec3f(20.197, 12.553, 5.383);
@if(CIE_D50) const YELLOW_XYZ: vec3f = vec3f(60.014, 60.949, 7.554);
@else const YELLOW_XYZ: vec3f = vec3f(55.779, 59.612, 9.448);
@if(CIE_D50) const MAGENTA_XYZ: vec3f = vec3f(32.305, 20.971, 23.891);
@else const MAGENTA_XYZ: vec3f = vec3f(30.689, 20.117, 32.074);
@if(CIE_D50) const CYAN_XYZ: vec3f = vec3f(13.964, 19.428, 31.039);
@else const CYAN_XYZ: vec3f = vec3f(15.131, 20.357, 40.473);
@if(CIE_D50) const WHITE_XYZ: vec3f = vec3f(87.473, 90.892, 73.275);
@else const WHITE_XYZ: vec3f = vec3f(86.047, 90.868, 96.433);
@if(CIE_D50) const NEUTRAL_80_XYZ: vec3f = vec3f(57.342, 59.788, 49.481);
@else const NEUTRAL_80_XYZ: vec3f = vec3f(56.562, 59.821, 65.218);
@if(CIE_D50) const NEUTRAL_65_XYZ: vec3f = vec3f(35.589, 37.181, 30.911);
@else const NEUTRAL_65_XYZ: vec3f = vec3f(35.144, 37.209, 40.764);
@if(CIE_D50) const NEUTRAL_50_XYZ: vec3f = vec3f(18.752, 19.493, 16.152);
@else const NEUTRAL_50_XYZ: vec3f = vec3f(18.505, 19.497, 21.306);
@if(CIE_D50) const NEUTRAL_35_XYZ: vec3f = vec3f(8.833, 9.223, 7.82);
@else const NEUTRAL_35_XYZ: vec3f = vec3f(8.737, 9.233, 10.323);
@if(CIE_D50) const BLACK_XYZ: vec3f = vec3f(3.225, 3.34, 2.822);
@else const BLACK_XYZ: vec3f = vec3f(3.185, 3.342, 3.727);

// LAB color space (D50 vs D65 illuminant)
@if(CIE_D50) const DARK_SKIN_LAB: vec3f = vec3f(39.4, 13.26, 14.44);
@else const DARK_SKIN_LAB: vec3f = vec3f(38.96, 11.7, 13.73);
@if(CIE_D50) const LIGHT_SKIN_LAB: vec3f = vec3f(65.81, 19.06, 17.15);
@else const LIGHT_SKIN_LAB: vec3f = vec3f(65.28, 15.68, 16.94);
@if(CIE_D50) const BLUE_SKY_LAB: vec3f = vec3f(50.29, -3.86, -22.11);
@else const BLUE_SKY_LAB: vec3f = vec3f(50.73, -0.92, -21.57);
@if(CIE_D50) const FOLIAGE_LAB: vec3f = vec3f(43.92, -13.73, 22.33);
@else const FOLIAGE_LAB: vec3f = vec3f(43.92, -15.31, 22.61);
@if(CIE_D50) const BLUE_FLOWER_LAB: vec3f = vec3f(55.77, 9.33, -24.86);
@else const BLUE_FLOWER_LAB: vec3f = vec3f(55.98, 11.87, -24.81);
@if(CIE_D50) const BLUISH_GREEN_LAB: vec3f = vec3f(70.99, -33.01, -0.87);
@else const BLUISH_GREEN_LAB: vec3f = vec3f(71.63, -32.29, 0.96);
@if(CIE_D50) const ORANGE_LAB: vec3f = vec3f(63.39, 35.03, 57.69);
@else const ORANGE_LAB: vec3f = vec3f(62.11, 31.79, 55.83);
@if(CIE_D50) const PURPLISH_BLUE_LAB: vec3f = vec3f(40.57, 11.01, -45.8);
@else const PURPLISH_BLUE_LAB: vec3f = vec3f(41.23, 17.51, -45.0);
@if(CIE_D50) const MODERATE_RED_LAB: vec3f = vec3f(52.25, 48.2, 16.989);
@else const MODERATE_RED_LAB: vec3f = vec3f(50.98, 45.91, 14.6);
@if(CIE_D50) const PURPLE_LAB: vec3f = vec3f(31.04, 22.31, -21.03);
@else const PURPLE_LAB: vec3f = vec3f(30.87, 24.25, -22.06);
@if(CIE_D50) const YELLOW_GREEN_LAB: vec3f = vec3f(72.95, -24.35, 56.48);
@else const YELLOW_GREEN_LAB: vec3f = vec3f(72.87, -28.16, 57.78);
@if(CIE_D50) const ORANGE_YELLOW_LAB: vec3f = vec3f(72.52, 18.45, 68.16);
@else const ORANGE_YELLOW_LAB: vec3f = vec3f(71.51, 14.6, 66.93);
@if(CIE_D50) const BLUE_LAB: vec3f = vec3f(29.47, 15.59, -50.68);
@else const BLUE_LAB: vec3f = vec3f(30.27, 23.04, -49.59);
@if(CIE_D50) const GREEN_LAB: vec3f = vec3f(55.63, -38.46, 30.77);
@else const GREEN_LAB: vec3f = vec3f(56.02, -40.94, 32.75);
@if(CIE_D50) const RED_LAB: vec3f = vec3f(43.57, 51.47, 29.3);
@else const RED_LAB: vec3f = vec3f(42.08, 48.02, 26.74);
@if(CIE_D50) const YELLOW_LAB: vec3f = vec3f(82.35, 2.97, 79.44);
@else const YELLOW_LAB: vec3f = vec3f(81.63, -2.19, 79.78);
@if(CIE_D50) const MAGENTA_LAB: vec3f = vec3f(52.92, 50.21, -13.48);
@else const MAGENTA_LAB: vec3f = vec3f(51.97, 50.05, -15.89);
@if(CIE_D50) const CYAN_LAB: vec3f = vec3f(51.18, -27.02, -28.54);
@else const CYAN_LAB: vec3f = vec3f(52.24, -23.14, -26.15);
@if(CIE_D50) const WHITE_LAB: vec3f = vec3f(96.37, -0.31, 1.5);
@else const WHITE_LAB: vec3f = vec3f(96.36, -0.6, 1.65);
@if(CIE_D50) const NEUTRAL_80_LAB: vec3f = vec3f(81.72, -0.75, -0.16);
@else const NEUTRAL_80_LAB: vec3f = vec3f(81.74, -0.73, -0.07);
@if(CIE_D50) const NEUTRAL_65_LAB: vec3f = vec3f(67.41, -0.88, -0.36);
@else const NEUTRAL_65_LAB: vec3f = vec3f(67.43, -0.75, -0.29);
@if(CIE_D50) const NEUTRAL_50_LAB: vec3f = vec3f(51.26, -0.22, -0.16);
@else const NEUTRAL_50_LAB: vec3f = vec3f(51.26, -0.14, -0.14);
@if(CIE_D50) const NEUTRAL_35_LAB: vec3f = vec3f(36.41, -0.5, -0.82);
@else const NEUTRAL_35_LAB: vec3f = vec3f(36.43, -0.33, -0.8);
@if(CIE_D50) const BLACK_LAB: vec3f = vec3f(21.36, 0.07, -0.5);
@else const BLACK_LAB: vec3f = vec3f(21.36, 0.14, -0.51);

// LCH color space (D50 vs D65 illuminant)
@if(CIE_D50) const DARK_SKIN_LCH: vec3f = vec3f(39.4, 19.61, 47.45);
@else const DARK_SKIN_LCH: vec3f = vec3f(38.96, 18.04, 49.57);
@if(CIE_D50) const LIGHT_SKIN_LCH: vec3f = vec3f(65.81, 25.64, 41.99);
@else const LIGHT_SKIN_LCH: vec3f = vec3f(65.28, 23.08, 47.21);
@if(CIE_D50) const BLUE_SKY_LCH: vec3f = vec3f(50.29, 22.44, 260.08);
@else const BLUE_SKY_LCH: vec3f = vec3f(50.73, 21.59, 267.55);
@if(CIE_D50) const FOLIAGE_LCH: vec3f = vec3f(43.92, 26.22, 121.59);
@else const FOLIAGE_LCH: vec3f = vec3f(43.92, 27.31, 124.12);
@if(CIE_D50) const BLUE_FLOWER_LCH: vec3f = vec3f(55.77, 26.56, 290.56);
@else const BLUE_FLOWER_LCH: vec3f = vec3f(55.98, 27.5, 295.57);
@if(CIE_D50) const BLUISH_GREEN_LCH: vec3f = vec3f(70.99, 33.02, 181.51);
@else const BLUISH_GREEN_LCH: vec3f = vec3f(71.63, 32.3, 178.29);
@if(CIE_D50) const ORANGE_LCH: vec3f = vec3f(63.39, 67.49, 58.74);
@else const ORANGE_LCH: vec3f = vec3f(62.11, 64.24, 60.34);
@if(CIE_D50) const PURPLISH_BLUE_LCH: vec3f = vec3f(40.57, 47.1, 283.528);
@else const PURPLISH_BLUE_LCH: vec3f = vec3f(41.23, 48.29, 291.26);
@if(CIE_D50) const MODERATE_RED_LCH: vec3f = vec3f(52.25, 51.1, 19.4);
@else const MODERATE_RED_LCH: vec3f = vec3f(50.98, 48.17, 17.64);
@if(CIE_D50) const PURPLE_LCH: vec3f = vec3f(31.04, 30.66, 316.69);
@else const PURPLE_LCH: vec3f = vec3f(30.87, 32.78, 317.71);
@if(CIE_D50) const YELLOW_GREEN_LCH: vec3f = vec3f(72.95, 61.51, 113.32);
@else const YELLOW_GREEN_LCH: vec3f = vec3f(72.87, 64.28, 115.98);
@if(CIE_D50) const ORANGE_YELLOW_LCH: vec3f = vec3f(72.52, 70.62, 74.85);
@else const ORANGE_YELLOW_LCH: vec3f = vec3f(71.51, 68.5, 77.69);
@if(CIE_D50) const BLUE_LCH: vec3f = vec3f(29.47, 53.03, 287.1);
@else const BLUE_LCH: vec3f = vec3f(30.27, 54.68, 294.92);
@if(CIE_D50) const GREEN_LCH: vec3f = vec3f(55.63, 49.26, 141.34);
@else const GREEN_LCH: vec3f = vec3f(56.02, 52.43, 141.35);
@if(CIE_D50) const RED_LCH: vec3f = vec3f(43.57, 59.22, 29.65);
@else const RED_LCH: vec3f = vec3f(42.08, 54.96, 29.11);
@if(CIE_D50) const YELLOW_LCH: vec3f = vec3f(82.35, 79.49, 87.86);
@else const YELLOW_LCH: vec3f = vec3f(81.63, 79.81, 91.57);
@if(CIE_D50) const MAGENTA_LCH: vec3f = vec3f(52.92, 51.99, 344.97);
@else const MAGENTA_LCH: vec3f = vec3f(51.97, 52.51, 342.39);
@if(CIE_D50) const CYAN_LCH: vec3f = vec3f(51.18, 39.29, 226.57);
@else const CYAN_LCH: vec3f = vec3f(52.24, 34.92, 228.49);
@if(CIE_D50) const WHITE_LCH: vec3f = vec3f(96.37, 1.53, 101.57);
@else const WHITE_LCH: vec3f = vec3f(96.36, 1.76, 109.99);
@if(CIE_D50) const NEUTRAL_80_LCH: vec3f = vec3f(81.72, 0.77, 192.3);
@else const NEUTRAL_80_LCH: vec3f = vec3f(81.74, 0.74, 185.65);
@if(CIE_D50) const NEUTRAL_65_LCH: vec3f = vec3f(67.41, 0.95, 202.11);
@else const NEUTRAL_65_LCH: vec3f = vec3f(67.43, 0.81, 201.35);
@if(CIE_D50) const NEUTRAL_50_LCH: vec3f = vec3f(51.26, 0.28, 215.6);
@else const NEUTRAL_50_LCH: vec3f = vec3f(51.26, 0.2, 225.42);
@if(CIE_D50) const NEUTRAL_35_LCH: vec3f = vec3f(36.41, 0.96, 238.548);
@else const NEUTRAL_35_LCH: vec3f = vec3f(36.43, 0.87, 247.43);
@if(CIE_D50) const BLACK_LCH: vec3f = vec3f(21.36, 0.51, 278.09);
@else const BLACK_LCH: vec3f = vec3f(21.36, 0.53, 285.38);

fn macbeth(index: i32) -> vec3f {
    var rgb = array<vec3f, MACBETH_TOTAL>(
        DARK_SKIN, LIGHT_SKIN, BLUE_SKY, FOLIAGE, BLUE_FLOWER, BLUISH_GREEN,
        ORANGE, PURPLISH_BLUE, MODERATE_RED, PURPLE, YELLOW_GREEN, ORANGE_YELLOW,
        BLUE, GREEN, RED, YELLOW, MAGENTA, CYAN,
        WHITE, NEUTRAL_80, NEUTRAL_65, NEUTRAL_50, NEUTRAL_35, BLACK
    );
    let idx = index % i32(MACBETH_TOTAL);
    return rgb[idx];
}

fn macbethXYZ(index: i32) -> vec3f {
    var xyz = array<vec3f, MACBETH_TOTAL>(
        DARK_SKIN_XYZ, LIGHT_SKIN_XYZ, BLUE_SKY_XYZ, FOLIAGE_XYZ, BLUE_FLOWER_XYZ, BLUISH_GREEN_XYZ,
        ORANGE_XYZ, PURPLISH_BLUE_XYZ, MODERATE_RED_XYZ, PURPLE_XYZ, YELLOW_GREEN_XYZ, ORANGE_YELLOW_XYZ,
        BLUE_XYZ, GREEN_XYZ, RED_XYZ, YELLOW_XYZ, MAGENTA_XYZ, CYAN_XYZ,
        WHITE_XYZ, NEUTRAL_80_XYZ, NEUTRAL_65_XYZ, NEUTRAL_50_XYZ, NEUTRAL_35_XYZ, BLACK_XYZ
    );
    let idx = index % i32(MACBETH_TOTAL);
    return xyz[idx];
}

fn macbethLAB(index: i32) -> vec3f {
    var lab = array<vec3f, MACBETH_TOTAL>(
        DARK_SKIN_LAB, LIGHT_SKIN_LAB, BLUE_SKY_LAB, FOLIAGE_LAB, BLUE_FLOWER_LAB, BLUISH_GREEN_LAB,
        ORANGE_LAB, PURPLISH_BLUE_LAB, MODERATE_RED_LAB, PURPLE_LAB, YELLOW_GREEN_LAB, ORANGE_YELLOW_LAB,
        BLUE_LAB, GREEN_LAB, RED_LAB, YELLOW_LAB, MAGENTA_LAB, CYAN_LAB,
        WHITE_LAB, NEUTRAL_80_LAB, NEUTRAL_65_LAB, NEUTRAL_50_LAB, NEUTRAL_35_LAB, BLACK_LAB
    );
    let idx = index % i32(MACBETH_TOTAL);
    return lab[idx];
}

fn macbethLCH(index: i32) -> vec3f {
    var lch = array<vec3f, MACBETH_TOTAL>(
        DARK_SKIN_LCH, LIGHT_SKIN_LCH, BLUE_SKY_LCH, FOLIAGE_LCH, BLUE_FLOWER_LCH, BLUISH_GREEN_LCH,
        ORANGE_LCH, PURPLISH_BLUE_LCH, MODERATE_RED_LCH, PURPLE_LCH, YELLOW_GREEN_LCH, ORANGE_YELLOW_LCH,
        BLUE_LCH, GREEN_LCH, RED_LCH, YELLOW_LCH, MAGENTA_LCH, CYAN_LCH,
        WHITE_LCH, NEUTRAL_80_LCH, NEUTRAL_65_LCH, NEUTRAL_50_LCH, NEUTRAL_35_LCH, BLACK_LCH
    );
    let idx = index % i32(MACBETH_TOTAL);
    return lch[idx];
}
