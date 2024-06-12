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
#ifndef LOW_SAT_RED
#define LOW_SAT_RED vec3(0.824, 0.475, 0.459)
#endif

#ifndef LOW_SAT_RED_LAB
#define LOW_SAT_RED_LAB vec3(61.35, 34.81, 18.38)
#endif

// A2
// 0.071, 0.264
// sRGB: 216 	179 	90
#ifndef LOW_SAT_YELLOW
#define LOW_SAT_YELLOW vec3(0.847, 0.702, 0.353)
#endif

#ifndef LOW_SAT_YELLOW_LAB
#define LOW_SAT_YELLOW_LAB vec3(75.50, 5.84, 50.42)
#endif

// A3
// 0.071, 0.421
// sRGB: 127 	175 	120
#ifndef LOW_SAT_GREEN
#define LOW_SAT_GREEN vec3(0.498, 0.686, 0.471)
#endif

#ifndef LOW_SAT_GREEN_LAB
#define LOW_SAT_GREEN_LAB vec3(66.82, -25.1, 23.47)
#endif

// A4
// 0.071, 0.579
// sRGB: 66 	157 	179
#ifndef LOW_SAT_CYAN
#define LOW_SAT_CYAN vec3(0.259, 0.616, 0.702)
#endif

#ifndef LOW_SAT_CYAN_LAB
#define LOW_SAT_CYAN_LAB vec3(60.53, -22.6, -20.40)
#endif

// A5
// 0.071, 0.736
// sRGB: 116 	147 	194
#ifndef LOW_SAT_BLUE
#define LOW_SAT_BLUE vec3(0.455, 0.576, 0.761)
#endif

#ifndef LOW_SAT_BLUE_LAB
#define LOW_SAT_BLUE_LAB vec3(59.66, -2.03, -28.46)
#endif

// A6
// 0.071, 0.893
// sRGB: 190 	121 	154
#ifndef LOW_SAT_MAGENTA
#define LOW_SAT_MAGENTA vec3(0.745, 0.475, 0.604)
#endif
#ifndef LOW_SAT_MAGENTA_LAB
#define LOW_SAT_MAGENTA_LAB vec3(59.15, 30.83, -5.72)
#endif

// B1
// 0.175, 0.107
// sRGB: 218 	203 	201
#ifndef RED_TINT_10
#define RED_TINT_10 vec3(0.855, 0.796, 0.788)
#endif

#ifndef RED_TINT_10_LAB
#define RED_TINT_10_LAB vec3(82.68, 5.03, 3.02)
#endif

// B2
// 0.175, 0.264
// sRGB: 203 	205 	196
#ifndef GREEN_TINT_10
#define GREEN_TINT_10 vec3(0.796, 0.804, 0.769)
#endif

#ifndef GREEN_TINT_10_LAB
#define GREEN_TINT_10_LAB vec3(82.25, -2.42, 3.78)
#endif

// B3
// 0.175, 0.421
// sRGB: 206 	203 	208
#ifndef BLUE_TINT_10
#define BLUE_TINT_10 vec3(0.808, 0.796, 0.816)
#endif

#ifndef BLUE_TINT_10_LAB
#define BLUE_TINT_10_LAB vec3(82.29, 2.20, -2.04)
#endif

// B4
// 0.175, 0.579
// sRGB: 66 	57 	58
#ifndef RED_TONE_90
#define RED_TONE_90 vec3(0.259, 0.224, 0.227)
#endif

#ifndef RED_TONE_90_LAB
#define RED_TONE_90_LAB vec3(24.89, 4.43, 0.78)
#endif

// B5
// 0.175, 0.736
// sRGB: 54 	61 	56
#ifndef GREEN_TONE_90
#define GREEN_TONE_90 vec3(0.212, 0.239, 0.220)
#endif

#ifndef GREEN_TONE_90_LAB
#define GREEN_TONE_90_LAB vec3(24.89, 4.43, 0.78)
#endif

// B6
// 0.175, 0.893
// sRGB: 63 	60 	69
#ifndef BLUE_TONE_90
#define BLUE_TONE_90 vec3(0.247, 0.235, 0.271)
#endif

#ifndef BLUE_TONE_90_LAB
#define BLUE_TONE_90_LAB vec3(24.89, 4.43, 0.78)
#endif

// C1
// 0.279, 0.107
// sRGB: 237 	206 	186
#ifndef LIGHTEST_SKIN
#define LIGHTEST_SKIN vec3(0.929, 0.808, 0.729)
#endif

#ifndef LIGHTEST_SKIN_LAB
#define LIGHTEST_SKIN_LAB vec3(85.42, 9.41, 14.49)
#endif

// C2
// 0.279, 0.264
// sRGB: 211 	175 	133
#ifndef LIGHTER_SKIN
#define LIGHTER_SKIN vec3(0.827, 0.686, 0.522)
#endif

#ifndef LIGHTER_SKIN_LAB
#define LIGHTER_SKIN_LAB vec3(74.28, 9.05, 27.21)
#endif

// C3
// 0.279, 0.421
// sRGB: 193 	149 	91
#ifndef MODERATE_SKIN
#define MODERATE_SKIN vec3(0.757, 0.584, 0.357)
#endif

#ifndef MODERATE_SKIN_LAB
#define MODERATE_SKIN_LAB vec3(64.57, 12.39, 37.24)
#endif

// C4
// 0.279, 0.579
// sRGB: 139 	93 	61
#ifndef MEDIUM_SKIN
#define MEDIUM_SKIN vec3(0.545, 0.365, 0.239)
#endif

#ifndef MEDIUM_SKIN_LAB
#define MEDIUM_SKIN_LAB vec3(44.49, 17.23, 26.24)
#endif

// C5
// 0.279, 0.736
// sRGB: 74 	55 	46
#ifndef DEEP_SKIN
#define DEEP_SKIN vec3(0.290, 0.216, 0.180)
#endif

#ifndef DEEP_SKIN_LAB
#define DEEP_SKIN_LAB vec3(25.29, 7.95, 8.87)
#endif

// C6
// 0.279, 0.893
// sRGB: 57 	54 	56
#ifndef GRAY_95
#define GRAY_95 vec3(0.224, 0.212, 0.220)
#endif

#ifndef GRAY_95_LAB
#define GRAY_95_LAB vec3(22.67, 2.11, -1.10)
#endif

// D1
// 0.384, 0.107
// sRGB: 241 	233 	229
#ifndef GRAY_05
#define GRAY_05 vec3(0.945, 0.914, 0.898)
#endif

#ifndef GRAY_05_LAB
#define GRAY_05_LAB vec3(90.31, 0.39, 1.09)
#endif

// D2
// 0.384, 0.264
// sRGB: 229 	222 	220
#ifndef GRAY_10
#define GRAY_10 vec3(0.898, 0.871, 0.863)
#endif

#ifndef GRAY_10_LAB
#define GRAY_10_LAB vec3(88.85, 1.59, 2.27)
#endif

// D3
// 0.384, 0.421
//sRGB 182 	178 	176
#ifndef GRAY_30
#define GRAY_30 vec3(0.714, 0.698, 0.690)
#endif

#ifndef GRAY_30_LAB
#define GRAY_30_LAB vec3(71.42, 0.99, 1.89)
#endif

// D4
// 0.384, 0.579
// sRGB: 139 	136 	135
#ifndef GRAY_50
#define GRAY_50 vec3(0.545, 0.533, 0.529)
#endif

#ifndef GRAY_50_LAB
#define GRAY_50_LAB vec3(55.89, 0.57, 1.19)
#endif

// D5
// 0.384, 0.736
// sRGB: 100 	99 	97
#ifndef GRAY_70
#define GRAY_70 vec3(0.392, 0.388, 0.380)
#endif

#ifndef GRAY_70_LAB
#define GRAY_70_LAB vec3(41.57, 0.24, 1.45)
#endif

// D6
// 0.384, 0.893
// sRGB: 63 	61 	62
#ifndef GRAY_90
#define GRAY_90 vec3(0.247, 0.239, 0.243)
#endif

#ifndef GRAY_90_LAB
#define GRAY_90_LAB vec3(25.65, 1.24, 0.05)
#endif

// E1
// 0.616, 0.107
// sRGB: 249, 242, 238
#ifndef CARD_WHITE
#define CARD_WHITE vec3(0.976, 0.949, 0.933)
#endif

#ifndef CARD_WHITE_LAB
#define CARD_WHITE_LAB vec3(95.99, 0.39, 1.09)
#endif

// E2
// 0.616, 0.264
// sRGB: 202, 198, 195
#ifndef GRAY_20
#define GRAY_20 vec3(0.792, 0.777, 0.765)
#endif

#ifndef GRAY_20_LAB
#define GRAY_20_LAB vec3(79.99, 1.17, 2.05)
#endif

// E3
// 0.616, 0.421
// sRGB: 161, 157, 154
#ifndef GRAY_40
#define GRAY_40 vec3(0.631, 0.616, 0.604)
#endif

#ifndef GRAY_40_LAB
#define GRAY_40_LAB vec3(65.52, 0.69, 1.86)
#endif


// E4
// 0.616, 0.579
// sRGB: 122, 118, 116
#ifndef GRAY_60
#define GRAY_60 vec3(0.478, 0.463, 0.455)
#endif

#ifndef GRAY_60_LAB
#define GRAY_60_LAB vec3(49.62, 0.58, 1.56)
#endif

// E5
// 0.616, 0.736
// sRGB: 80, 80, 78
#ifndef GRAY_80
#define GRAY_80 vec3(0.314, 0.314, 0.306)
#endif

#ifndef GRAY_80_LAB
#define GRAY_80_LAB vec3(33.55, 0.35, 1.40)
#endif

// E6
// 0.616, 0.893
// sRGB: 43, 41, 43
#ifndef CARD_BLACK
#define CARD_BLACK vec3(0.169, 0.161, 0.169)
#endif

#ifndef CARD_BLACK_LAB
#define CARD_BLACK_LAB vec3(16.91, 1.43, -0.81 )
#endif

// F1
// 0.721, 0.107
// sRGB: 0, 127, 159
#ifndef PRIMARY_CYAN
#define PRIMARY_CYAN vec3(0.000, 0.498, 0.623)
#endif

#ifndef PRIMARY_CYAN_LAB
#define PRIMARY_CYAN_LAB vec3(47.12, -32.50, -28.75)
#endif


// F2
//  0.721, 0.264
// sRGB: 192, 75, 145
#ifndef PRIMARY_MAGENTA
#define PRIMARY_MAGENTA vec3(0.753, 0.294, 0.569)
#endif

#ifndef PRIMARY_MAGENTA_LAB
#define PRIMARY_MAGENTA_LAB vec3(50.49, 53.45, -13.55)
#endif

// F3
// 0.721, 0.421
// sRGB: 245, 205, 0
#ifndef PRIMARY_YELLOW
#define PRIMARY_YELLOW vec3(0.961, 0.804, 0.000)
#endif 

#ifndef PRIMARY_YELLOW_LAB
#define PRIMARY_YELLOW_LAB vec3(83.61, 3.36, 87.02)
#endif

// F4
// 0.721, 0.579
// sRGB: 186, 26, 51
#ifndef PRIMARY_RED
#define PRIMARY_RED vec3(0.729, 0.102, 0.200)
#endif

#ifndef PRIMARY_RED_LAB
#define PRIMARY_RED_LAB vec3(41.05, 60.75, 31.17)
#endif

// F5
// 0.721, 0.736
// sRGB: 57, 146, 64
#ifndef PRIMARY_GREEN
#define PRIMARY_GREEN vec3(0.224, 0.573, 0.251)
#endif

#ifndef PRIMARY_GREEN_LAB
#define PRIMARY_GREEN_LAB vec3(54.14, -40.80, 34.75)
#endif

// F6
// 0.721, 0.893
// sRGB: 25, 55, 135
#ifndef PRIMARY_BLUE
#define PRIMARY_BLUE vec3(0.098, 0.216, 0.529)
#endif

#ifndef PRIMARY_BLUE_LAB
#define PRIMARY_BLUE_LAB vec3(24.75, 13.78, -49.48)
#endif

// G1
// 0.825, 0.107
// sRGB: 222, 118, 32
#ifndef PRIMARY_ORANGE
#define PRIMARY_ORANGE vec3(0.871, 0.463, 0.125)
#endif

#ifndef PRIMARY_ORANGE_LAB
#define PRIMARY_ORANGE_LAB vec3(60.94, 38.21, 61.31)
#endif

// G2
// 0.825, 0.26
// sRGB: 58, 89, 160
#ifndef BLUEPRINT
#define BLUEPRINT vec3(0.227, 0.349, 0.627)
#endif

#ifndef BLUEPRINT_LAB
#define BLUEPRINT_LAB vec3(37.80, 7.30, -43.04)
#endif

// G3
// 0.825, 0.421
// sRGB: 195, 79, 95
#ifndef PINK
#define PINK vec3(0.765, 0.310, 0.373)
#endif

#ifndef PINK_LAB
#define PINK_LAB vec3(49.81, 48.50, 15.76)
#endif

// G4
// 0.825, 0.57
// sRGB: 83, 58, 106
#ifndef VIOLET
#define VIOLET vec3(0.325, 0.227, 0.416)
#endif

#ifndef VIOLET_LAB
#define VIOLET_LAB vec3(28.88, 19.36, -24.48)
#endif

// G5
// 0.825, 0.73
// sRGB: 157, 188, 54
#ifndef APPLE_GREEN
#define APPLE_GREEN vec3(0.616, 0.737, 0.212)
#endif

#ifndef APPLE_GREEN_LAB
#define APPLE_GREEN_LAB vec3(72.45, -23.60, 60.47)
#endif

// G6
// 0.825, 0.893
// sRGB: 238, 158, 25
#ifndef SUNFLOWER
#define SUNFLOWER vec3(0.933, 0.620, 0.098)
#endif

#ifndef SUNFLOWER_LAB
#define SUNFLOWER_LAB vec3(71.65, 23.74, 72.28)
#endif

// H1
// 0.929, 0.107
// sRGB: 98, 187, 166
#ifndef AQUA
#define AQUA vec3(0.384, 0.733, 0.651)
#endif

#ifndef AQUA_LAB
#define AQUA_LAB vec3(70.19, -31.90, 1.98)
#endif

// H2
// 0.929, 0.264
// sRGB: 126, 125, 174
#ifndef LAVANDER
#define LAVANDER vec3(0.494, 0.490, 0.682)
#endif

#ifndef LAVANDER_LAB
#define LAVANDER_LAB vec3(54.38, 8.84, -25.71)
#endif

// H3
// 0.929, 0.421
// sRGB: 82, 106, 60
#ifndef EVERGREEN
#define EVERGREEN vec3(0.322, 0.423, 0.247)
#endif

#ifndef EVERGREEN_LAB
#define EVERGREEN_LAB vec3(42.03, -15.80, 22.93)
#endif

// H4
// 0.929, 0.579
// sRGB: 87, 120, 155
#ifndef STEEL_BLUE
#define STEEL_BLUE vec3(0.341, 0.467, 0.603)
#endif

#ifndef STEEL_BLUE_LAB
#define STEEL_BLUE_LAB vec3(48.82, -5.11, -23.08)
#endif

// H5
// 0.929, 0.736 
// sRGB: 197, 145, 125
#ifndef CLASSIC_LIGHT_SKIN
#define CLASSIC_LIGHT_SKIN vec3(0.769, 0.557, 0.494)
#endif

#ifndef CLASSIC_LIGHT_SKIN_LAB
#define CLASSIC_LIGHT_SKIN_LAB vec3(65.10, 18.14, 18.68)
#endif

// H6
// 0.929, 0.893 
// sRGB: 112, 76, 60
#ifndef CLASSIC_DARK_SKIN
#define CLASSIC_DARK_SKIN vec3(0.439, 0.302, 0.247)
#endif

#ifndef CLASSIC_DARK_SKIN_LAB
#define CLASSIC_DARK_SKIN_LAB vec3(36.13, 14.15, 15.78)
#endif

#ifndef FNC_PALETTE_SPYDER
#define FNC_PALETTE_SPYDER

vec3 spyder (const int index) {
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

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 48; i++)
        if (i == index) return colors[i];
    #else
    return colors[index];
    #endif
}

vec3 spyderLAB (const int index) {
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

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 48; i++)
        if (i == index) return colors[i];
    #else
    return colors[index];
    #endif
}

vec3 spyderA (const int index) { return spyder(index);}
vec3 spyderB (const int index) { return spyder(index + 24);}
vec3 spyderALAB (const int index) { return spyderLAB(index);}
vec3 spyderBLAB (const int index) { return spyderLAB(index + 24);}

#endif