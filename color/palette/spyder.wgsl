/*
contributors: Patricio Gonzalez Vivo
description: |
    SpyderChecker values from:
    - https://www.northlight-images.co.uk/datacolor-spydercheckr-colour-test-card-review/
    - https://www.bartneck.de/2017/10/24/patch-color-definitions-for-datacolor-spydercheckr-48/
use:
    - <vec3> spyder (<int> index)
    - <vec3> spyderA (<int> index)
    - <vec3> spyderB (<int> index)
    - <vec3> spyderLAB (<int> index)
    - <vec3> spyderALAB (<int> index)
    - <vec3> spyderBLAB (<int> index)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_colorChecker.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// A1
// 0.071, 0.107
// sRGB: 210, 121, 117
// #define LOW_SAT_RED vec3(0.824, 0.475, 0.459)

// #define LOW_SAT_RED_LAB vec3(61.35, 34.81, 18.38)

// A2
// 0.071, 0.264
// sRGB: 216 	179 	90
// #define LOW_SAT_YELLOW vec3(0.847, 0.702, 0.353)

// #define LOW_SAT_YELLOW_LAB vec3(75.50, 5.84, 50.42)

// A3
// 0.071, 0.421
// sRGB: 127 	175 	120
// #define LOW_SAT_GREEN vec3(0.498, 0.686, 0.471)

// #define LOW_SAT_GREEN_LAB vec3(66.82, -25.1, 23.47)

// A4
// 0.071, 0.579
// sRGB: 66 	157 	179
// #define LOW_SAT_CYAN vec3(0.259, 0.616, 0.702)

// #define LOW_SAT_CYAN_LAB vec3(60.53, -22.6, -20.40)

// A5
// 0.071, 0.736
// sRGB: 116 	147 	194
// #define LOW_SAT_BLUE vec3(0.455, 0.576, 0.761)

// #define LOW_SAT_BLUE_LAB vec3(59.66, -2.03, -28.46)

// A6
// 0.071, 0.893
// sRGB: 190 	121 	154
// #define LOW_SAT_MAGENTA vec3(0.745, 0.475, 0.604)
// #define LOW_SAT_MAGENTA_LAB vec3(59.15, 30.83, -5.72)

// B1
// 0.175, 0.107
// sRGB: 218 	203 	201
// #define RED_TINT_10 vec3(0.855, 0.796, 0.788)

// #define RED_TINT_10_LAB vec3(82.68, 5.03, 3.02)

// B2
// 0.175, 0.264
// sRGB: 203 	205 	196
// #define GREEN_TINT_10 vec3(0.796, 0.804, 0.769)

// #define GREEN_TINT_10_LAB vec3(82.25, -2.42, 3.78)

// B3
// 0.175, 0.421
// sRGB: 206 	203 	208
// #define BLUE_TINT_10 vec3(0.808, 0.796, 0.816)

// #define BLUE_TINT_10_LAB vec3(82.29, 2.20, -2.04)

// B4
// 0.175, 0.579
// sRGB: 66 	57 	58
// #define RED_TONE_90 vec3(0.259, 0.224, 0.227)

// #define RED_TONE_90_LAB vec3(24.89, 4.43, 0.78)

// B5
// 0.175, 0.736
// sRGB: 54 	61 	56
// #define GREEN_TONE_90 vec3(0.212, 0.239, 0.220)

// #define GREEN_TONE_90_LAB vec3(24.89, 4.43, 0.78)

// B6
// 0.175, 0.893
// sRGB: 63 	60 	69
// #define BLUE_TONE_90 vec3(0.247, 0.235, 0.271)

// #define BLUE_TONE_90_LAB vec3(24.89, 4.43, 0.78)

// C1
// 0.279, 0.107
// sRGB: 237 	206 	186
// #define LIGHTEST_SKIN vec3(0.929, 0.808, 0.729)

// #define LIGHTEST_SKIN_LAB vec3(85.42, 9.41, 14.49)

// C2
// 0.279, 0.264
// sRGB: 211 	175 	133
// #define LIGHTER_SKIN vec3(0.827, 0.686, 0.522)

// #define LIGHTER_SKIN_LAB vec3(74.28, 9.05, 27.21)

// C3
// 0.279, 0.421
// sRGB: 193 	149 	91
// #define MODERATE_SKIN vec3(0.757, 0.584, 0.357)

// #define MODERATE_SKIN_LAB vec3(64.57, 12.39, 37.24)

// C4
// 0.279, 0.579
// sRGB: 139 	93 	61
// #define MEDIUM_SKIN vec3(0.545, 0.365, 0.239)

// #define MEDIUM_SKIN_LAB vec3(44.49, 17.23, 26.24)

// C5
// 0.279, 0.736
// sRGB: 74 	55 	46
// #define DEEP_SKIN vec3(0.290, 0.216, 0.180)

// #define DEEP_SKIN_LAB vec3(25.29, 7.95, 8.87)

// C6
// 0.279, 0.893
// sRGB: 57 	54 	56
// #define GRAY_95 vec3(0.224, 0.212, 0.220)

// #define GRAY_95_LAB vec3(22.67, 2.11, -1.10)

// D1
// 0.384, 0.107
// sRGB: 241 	233 	229
// #define GRAY_05 vec3(0.945, 0.914, 0.898)

// #define GRAY_05_LAB vec3(90.31, 0.39, 1.09)

// D2
// 0.384, 0.264
// sRGB: 229 	222 	220
// #define GRAY_10 vec3(0.898, 0.871, 0.863)

// #define GRAY_10_LAB vec3(88.85, 1.59, 2.27)

// D3
// 0.384, 0.421
//sRGB 182 	178 	176
// #define GRAY_30 vec3(0.714, 0.698, 0.690)

// #define GRAY_30_LAB vec3(71.42, 0.99, 1.89)

// D4
// 0.384, 0.579
// sRGB: 139 	136 	135
// #define GRAY_50 vec3(0.545, 0.533, 0.529)

// #define GRAY_50_LAB vec3(55.89, 0.57, 1.19)

// D5
// 0.384, 0.736
// sRGB: 100 	99 	97
// #define GRAY_70 vec3(0.392, 0.388, 0.380)

// #define GRAY_70_LAB vec3(41.57, 0.24, 1.45)

// D6
// 0.384, 0.893
// sRGB: 63 	61 	62
// #define GRAY_90 vec3(0.247, 0.239, 0.243)

// #define GRAY_90_LAB vec3(25.65, 1.24, 0.05)

// E1
// 0.616, 0.107
// sRGB: 249, 242, 238
// #define CARD_WHITE vec3(0.976, 0.949, 0.933)

// #define CARD_WHITE_LAB vec3(95.99, 0.39, 1.09)

// E2
// 0.616, 0.264
// sRGB: 202, 198, 195
// #define GRAY_20 vec3(0.792, 0.777, 0.765)

// #define GRAY_20_LAB vec3(79.99, 1.17, 2.05)

// E3
// 0.616, 0.421
// sRGB: 161, 157, 154
// #define GRAY_40 vec3(0.631, 0.616, 0.604)

// #define GRAY_40_LAB vec3(65.52, 0.69, 1.86)

// E4
// 0.616, 0.579
// sRGB: 122, 118, 116
// #define GRAY_60 vec3(0.478, 0.463, 0.455)

// #define GRAY_60_LAB vec3(49.62, 0.58, 1.56)

// E5
// 0.616, 0.736
// sRGB: 80, 80, 78
// #define GRAY_80 vec3(0.314, 0.314, 0.306)

// #define GRAY_80_LAB vec3(33.55, 0.35, 1.40)

// E6
// 0.616, 0.893
// sRGB: 43, 41, 43
// #define CARD_BLACK vec3(0.169, 0.161, 0.169)

// #define CARD_BLACK_LAB vec3(16.91, 1.43, -0.81 )

// F1
// 0.721, 0.107
// sRGB: 0, 127, 159
// #define PRIMARY_CYAN vec3(0.000, 0.498, 0.623)

// #define PRIMARY_CYAN_LAB vec3(47.12, -32.50, -28.75)

// F2
//  0.721, 0.264
// sRGB: 192, 75, 145
// #define PRIMARY_MAGENTA vec3(0.753, 0.294, 0.569)

// #define PRIMARY_MAGENTA_LAB vec3(50.49, 53.45, -13.55)

// F3
// 0.721, 0.421
// sRGB: 245, 205, 0
// #define PRIMARY_YELLOW vec3(0.961, 0.804, 0.000)

// #define PRIMARY_YELLOW_LAB vec3(83.61, 3.36, 87.02)

// F4
// 0.721, 0.579
// sRGB: 186, 26, 51
// #define PRIMARY_RED vec3(0.729, 0.102, 0.200)

// #define PRIMARY_RED_LAB vec3(41.05, 60.75, 31.17)

// F5
// 0.721, 0.736
// sRGB: 57, 146, 64
// #define PRIMARY_GREEN vec3(0.224, 0.573, 0.251)

// #define PRIMARY_GREEN_LAB vec3(54.14, -40.80, 34.75)

// F6
// 0.721, 0.893
// sRGB: 25, 55, 135
// #define PRIMARY_BLUE vec3(0.098, 0.216, 0.529)

// #define PRIMARY_BLUE_LAB vec3(24.75, 13.78, -49.48)

// G1
// 0.825, 0.107
// sRGB: 222, 118, 32
// #define PRIMARY_ORANGE vec3(0.871, 0.463, 0.125)

// #define PRIMARY_ORANGE_LAB vec3(60.94, 38.21, 61.31)

// G2
// 0.825, 0.26
// sRGB: 58, 89, 160
// #define BLUEPRINT vec3(0.227, 0.349, 0.627)

// #define BLUEPRINT_LAB vec3(37.80, 7.30, -43.04)

// G3
// 0.825, 0.421
// sRGB: 195, 79, 95
// #define PINK vec3(0.765, 0.310, 0.373)

// #define PINK_LAB vec3(49.81, 48.50, 15.76)

// G4
// 0.825, 0.57
// sRGB: 83, 58, 106
// #define VIOLET vec3(0.325, 0.227, 0.416)

// #define VIOLET_LAB vec3(28.88, 19.36, -24.48)

// G5
// 0.825, 0.73
// sRGB: 157, 188, 54
// #define APPLE_GREEN vec3(0.616, 0.737, 0.212)

// #define APPLE_GREEN_LAB vec3(72.45, -23.60, 60.47)

// G6
// 0.825, 0.893
// sRGB: 238, 158, 25
// #define SUNFLOWER vec3(0.933, 0.620, 0.098)

// #define SUNFLOWER_LAB vec3(71.65, 23.74, 72.28)

// H1
// 0.929, 0.107
// sRGB: 98, 187, 166
// #define AQUA vec3(0.384, 0.733, 0.651)

// #define AQUA_LAB vec3(70.19, -31.90, 1.98)

// H2
// 0.929, 0.264
// sRGB: 126, 125, 174
// #define LAVANDER vec3(0.494, 0.490, 0.682)

// #define LAVANDER_LAB vec3(54.38, 8.84, -25.71)

// H3
// 0.929, 0.421
// sRGB: 82, 106, 60
// #define EVERGREEN vec3(0.322, 0.423, 0.247)

// #define EVERGREEN_LAB vec3(42.03, -15.80, 22.93)

// H4
// 0.929, 0.579
// sRGB: 87, 120, 155
// #define STEEL_BLUE vec3(0.341, 0.467, 0.603)

// #define STEEL_BLUE_LAB vec3(48.82, -5.11, -23.08)

// H5
// 0.929, 0.736 
// sRGB: 197, 145, 125
// #define CLASSIC_LIGHT_SKIN vec3(0.769, 0.557, 0.494)

// #define CLASSIC_LIGHT_SKIN_LAB vec3(65.10, 18.14, 18.68)

// H6
// 0.929, 0.893 
// sRGB: 112, 76, 60
// #define CLASSIC_DARK_SKIN vec3(0.439, 0.302, 0.247)

// #define CLASSIC_DARK_SKIN_LAB vec3(36.13, 14.15, 15.78)

fn spyder(index: i32) -> vec3f {
    vec3 colors[48];
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

    colors[24] = AQUA;
    colors[25] = LAVANDER;
    colors[26] = EVERGREEN;
    colors[27] = STEEL_BLUE;
    colors[28] = CLASSIC_LIGHT_SKIN;
    colors[29] = CLASSIC_DARK_SKIN;

    colors[30] = PRIMARY_ORANGE;
    colors[31] = BLUEPRINT;
    colors[32] = PINK;
    colors[33] = VIOLET;
    colors[34] = APPLE_GREEN;
    colors[35] = SUNFLOWER;
    
    colors[36] = PRIMARY_CYAN;
    colors[37] = PRIMARY_MAGENTA;
    colors[38] = PRIMARY_YELLOW;
    colors[39] = PRIMARY_RED;
    colors[40] = PRIMARY_GREEN;
    colors[41] = PRIMARY_BLUE;

    colors[42] = CARD_WHITE;
    colors[43] = GRAY_20;
    colors[44] = GRAY_40;
    colors[45] = GRAY_60;
    colors[46] = GRAY_80;
    colors[47] = CARD_BLACK;

    for (int i = 0; i < 48; i++)
        if (i == index) return colors[i];
    return colors[index];
}

fn spyderLAB(index: i32) -> vec3f {
    vec3 colors[48];
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

    colors[24] = AQUA_LAB;
    colors[25] = LAVANDER_LAB;
    colors[26] = EVERGREEN_LAB;
    colors[27] = STEEL_BLUE_LAB;
    colors[28] = CLASSIC_LIGHT_SKIN_LAB;
    colors[29] = CLASSIC_DARK_SKIN_LAB;

    colors[30] = PRIMARY_ORANGE_LAB;
    colors[31] = BLUEPRINT_LAB;
    colors[32] = PINK_LAB;
    colors[33] = VIOLET_LAB;
    colors[34] = APPLE_GREEN_LAB;
    colors[35] = SUNFLOWER_LAB;
    
    colors[36] = PRIMARY_CYAN_LAB;
    colors[37] = PRIMARY_MAGENTA_LAB;
    colors[38] = PRIMARY_YELLOW_LAB;
    colors[39] = PRIMARY_RED_LAB;
    colors[40] = PRIMARY_GREEN_LAB;
    colors[41] = PRIMARY_BLUE_LAB;

    colors[42] = CARD_WHITE_LAB;
    colors[43] = GRAY_20_LAB;
    colors[44] = GRAY_40_LAB;
    colors[45] = GRAY_60_LAB;
    colors[46] = GRAY_80_LAB;
    colors[47] = CARD_BLACK_LAB;

    for (int i = 0; i < 48; i++)
        if (i == index) return colors[i];
    return colors[index];
}

fn spyderA(index: i32) -> vec3f { return spyder(index);}
fn spyderB(index: i32) -> vec3f { return spyder(index + 24);}
fn spyderALAB(index: i32) -> vec3f { return spyderLAB(index);}
fn spyderBLAB(index: i32) -> vec3f { return spyderLAB(index + 24);}
