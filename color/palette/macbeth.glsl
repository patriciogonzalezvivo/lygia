/*
contributors: Patricio Gonzalez Vivo
description: | 
    MacBeth values from:
        - http://en.wikipedia.org/wiki/ColorChecker
        - http://kurtmunger.com/color_checkerid277.html
        - http://www.rags-int-inc.com/phototechstuff/macbethtarget/
        - https://babelcolor.com/index_htm_files/RGB%20Coordinates%20of%20the%20Macbeth%20ColorChecker.pdf
use:
    - <vec3> macbeth (<int> index)
    - <vec3> macbethXYZ (<int> index)
    - <vec3> macbethLAB (<int> index)
    - <vec3> macbethLCH (<int> index)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_colorChecker.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// Dark skin (3 YR 3.7/3.2)
#ifndef DARK_SKIN
// sRGB: 117	84	70
#define DARK_SKIN vec3(0.46017,0.33059,0.27477)
#endif

#ifndef DARK_SKIN_XYZ
#ifdef CIE_D50
#define DARK_SKIN_XYZ vec3(12.354,10.896,5.498)
#else
#define DARK_SKIN_XYZ vec3(11.684,10.637,7.242)
#endif
#endif

#ifndef DARK_SKIN_LAB
#ifdef CIE_D50
#define DARK_SKIN_LAB vec3(39.4,13.26,14.44)
#else
#define DARK_SKIN_LAB vec3(38.96,11.7,13.73)
#endif
#endif

#ifndef DARK_SKIN_LCH
#ifdef CIE_D50
#define DARK_SKIN_LCH vec3(39.4,19.61,47.45)
#else
#define DARK_SKIN_LCH vec3(38.96,18.04,49.57)
#endif
#endif

#ifndef DARK_SKIN_XYY
#define DARK_SKIN_XYY vec3(0.400, 0.350, 10.1)
#endif

// Light skin (2.2 YR 6.47/4.1)
#ifndef LIGHT_SKIN
// sRGB: 196	147	129
#define LIGHT_SKIN vec3(0.769, 0.576, 0.506)
#endif

#ifndef LIGHT_SKIN_XYZ
#ifdef CIE_D50
#define LIGHT_SKIN_XYZ vec3(39.602,35.073,19.617)
#else
#define LIGHT_SKIN_XYZ vec3(37.29,34.404,25.454)
#endif
#endif

#ifndef LIGHT_SKIN_LAB
#ifdef CIE_D50
#define LIGHT_SKIN_LAB vec3(65.81,19.06,17.15)
#else
#define LIGHT_SKIN_LAB vec3(65.28,15.68,16.94)
#endif
#endif

#ifndef LIGHT_SKIN_LCH
#ifdef CIE_D50
#define LIGHT_SKIN_LCH vec3(65.81,25.64,41.99)
#else
#define LIGHT_SKIN_LCH vec3(65.28,23.08,47.21)
#endif
#endif

#ifndef LIGHT_SKIN_XYY
#define LIGHT_SKIN_XYY vec3(0.377, 0.345, 35.8)
#endif


// Blue sky	(4.3 PB 4.95/5.5)	
#ifndef BLUE_SKY
// sRGB: 98 122 157
#define BLUE_SKY vec3(0.356, 0.472, 0.609)
#endif

#ifndef BLUE_SKY_XYZ
#ifdef CIE_D50
#define BLUE_SKY_XYZ vec3(17.275,18.663,26.177)
#else
#define BLUE_SKY_XYZ vec3(17.916,19.032,34.703)
#endif
#endif

#ifndef BLUE_SKY_LAB
#ifdef CIE_D50
#define BLUE_SKY_LAB vec3(50.29,-3.86,-22.11)
#else
#define BLUE_SKY_LAB vec3(50.73,-0.92,-21.57)
#endif
#endif

#ifndef BLUE_SKY_LCH
#ifdef CIE_D50
#define BLUE_SKY_LCH vec3(50.29,22.44,260.08)
#else
#define BLUE_SKY_LCH vec3(50.73,21.59,267.55)
#endif
#endif

#ifndef BLUE_SKY_XYY
#define BLUE_SKY_XYY vec3(0.247, 0.251, 19.3)
#endif

// Foliage		(6.7 GY 4.2/4.1)	
#ifndef FOLIAGE
// sRGB: 91	109	65
#define FOLIAGE vec3(0.357, 0.427, 0.247)
#endif

#ifndef FOLIAGE_XYZ
#ifdef CIE_D50
#define FOLIAGE_XYZ vec3(11.284,13.786,5.479)
#else
#define FOLIAGE_XYZ vec3(10.906,13.784,7.155)
#endif
#endif

#ifndef FOLIAGE_LAB
#ifdef CIE_D50
#define FOLIAGE_LAB vec3(43.92,-13.73,22.33)
#else
#define FOLIAGE_LAB vec3(43.92,-15.31,22.61)
#endif
#endif

#ifndef FOLIAGE_LCH
#ifdef CIE_D50
#define FOLIAGE_LCH vec3(43.92,26.22,121.59)
#else
#define FOLIAGE_LCH vec3(43.92,27.31,124.12)
#endif
#endif

#ifndef FOLIAGE_XYY
#define FOLIAGE_XYY vec3(0.337, 0.422, 13.3)
#endif

// Blue flower	(9.7 PB 5.47/6.7)
#ifndef BLUE_FLOWER
// sRGB: 132	129	177
#define BLUE_FLOWER vec3(0.518, 0.506, 0.694)
#endif

#ifndef BLUE_FLOWER_XYZ
#ifdef CIE_D50
#define BLUE_FLOWER_XYZ vec3(24.969,23.688,33.857)
#else
#define BLUE_FLOWER_XYZ vec3(25.42,23.896,44.94)
#endif
#endif

#ifndef BLUE_FLOWER_LAB
#ifdef CIE_D50
#define BLUE_FLOWER_LAB vec3(55.77,9.33,-24.86)
#else
#define BLUE_FLOWER_LAB vec3(55.98,11.87,-24.81)
#endif
#endif

#ifndef BLUE_FLOWER_LCH
#ifdef CIE_D50
#define BLUE_FLOWER_LCH vec3(55.77,26.56,290.56)
#else
#define BLUE_FLOWER_LCH vec3(55.98,27.5,295.57)
#endif
#endif

#ifndef BLUE_FLOWER_XYY
#define BLUE_FLOWER_XYY vec3(0.265, 0.240, 24.3)
#endif

// Bluish green	(2.5 BG 7/6)
#ifndef BLUISH_GREEN
// sRGB: 98	191	172
#define BLUISH_GREEN vec3(0.384, 0.749, 0.675)
#endif

#ifndef BLUISH_GREEN_XYZ
#ifdef CIE_D50
#define BLUISH_GREEN_XYZ vec3(30.838,42.168,35.407)
#else
#define BLUISH_GREEN_XYZ vec3(31.342,43.112,46.048)
#endif
#endif

#ifndef BLUISH_GREEN_LAB
#ifdef CIE_D50
#define BLUISH_GREEN_LAB vec3(70.99,-33.01,-0.87)
#else
#define BLUISH_GREEN_LAB vec3(71.63,-32.29,0.96)
#endif
#endif

#ifndef BLUISH_GREEN_LCH
#ifdef CIE_D50
#define BLUISH_GREEN_LCH vec3(70.99,33.02,181.51)
#else
#define BLUISH_GREEN_LCH vec3(71.63,32.3,178.29)
#endif
#endif

#ifndef BLUISH_GREEN_XYY
#define BLUISH_GREEN_XYY vec3(0.261, 0.343, 43.1)
#endif

// Orange		(5 YR 6/11)
#ifndef ORANGE
// sRGB: 221	125	47
#define ORANGE vec3(0.867, 0.487, 0.184)
#endif

#ifndef ORANGE_XYZ
#ifdef CIE_D50
#define ORANGE_XYZ vec3(41.4,32.052,5.12)
#else
#define ORANGE_XYZ vec3(38.041,30.531,6.671)
#endif
#endif

#ifndef ORANGE_LAB
#ifdef CIE_D50
#define ORANGE_LAB vec3(63.39,35.03,57.69)
#else
#define ORANGE_LAB vec3(62.11,31.79,55.83)
#endif
#endif

#ifndef ORANGE_LCH
#ifdef CIE_D50
#define ORANGE_LCH vec3(63.39,67.49,58.74)
#else
#define ORANGE_LCH vec3(62.11,64.24,60.34)
#endif
#endif

#ifndef ORANGE_XYY
#define ORANGE_XYY vec3(0.506, 0.407, 30.1)
#endif

// Purplish blue	(7.5 PB 4/10.7)	
#ifndef PURPLISH_BLUE
// sRGB: 74	91	171
#define PURPLISH_BLUE vec3(0.290, 0.357, 0.671)
#endif

#ifndef PURPLISH_BLUE_XYZ
#ifdef CIE_D50
#define PURPLISH_BLUE_XYZ vec3(12.767,11.597,30.374)
#else
#define PURPLISH_BLUE_XYZ vec3(14.025,12.012,40.373)
#endif
#endif

#ifndef PURPLISH_BLUE_LAB
#ifdef CIE_D50
#define PURPLISH_BLUE_LAB vec3(40.57,11.01,-45.8)
#else
#define PURPLISH_BLUE_LAB vec3(41.23,17.51,-45.0)
#endif
#endif

#ifndef PURPLISH_BLUE_LCH
#ifdef CIE_D50
#define PURPLISH_BLUE_LCH vec3(40.57,47.1,283.528)
#else
#define PURPLISH_BLUE_LCH vec3(41.23,48.29,291.26)
#endif
#endif

#ifndef PURPLISH_BLUE_XYY
#define PURPLISH_BLUE_XYY vec3(0.211, 0.175, 12.0)
#endif

// Moderate red	(2.5 R 5/10)
#ifndef MODERATE_RED
// sRGB: 196	85	98
#define MODERATE_RED vec3(0.769, 0.333, 0.384)
#endif

#ifndef MODERATE_RED_XYZ
#ifdef CIE_D50
#define MODERATE_RED_XYZ vec3(30.96,20.368,10.532)
#else
#define MODERATE_RED_XYZ vec3(28.491,19.254,13.978)
#endif
#endif

#ifndef MODERATE_RED_LAB
#ifdef CIE_D50
#define MODERATE_RED_LAB vec3(52.25,48.2,16.989)
#else
#define MODERATE_RED_LAB vec3(50.98,45.91,14.6)
#endif
#endif

#ifndef MODERATE_RED_LCH
#ifdef CIE_D50
#define MODERATE_RED_LCH vec3(52.25,51.1,19.4)
#else
#define MODERATE_RED_LCH vec3(50.98,48.17,17.64)
#endif
#endif

#ifndef MODERATE_RED_XYY
#define MODERATE_RED_XYY vec3(0.453, 0.306, 19.8)
#endif

// Purple		(5 P 3/7)
#ifndef PURPLE
// sRGB: 93	59	107
#define PURPLE vec3(0.365, 0.231, 0.420)
#endif

#ifndef PURPLE_XYZ
#ifdef CIE_D50
#define PURPLE_XYZ vec3(8.795,6.668,10.99)
#else
#define PURPLE_XYZ vec3(8.81,6.597,14.816)
#endif
#endif

#ifndef PURPLE_LAB
#ifdef CIE_D50
#define PURPLE_LAB vec3(31.04,22.31,-21.03)
#else
#define PURPLE_LAB vec3(30.87,24.25,-22.06)
#endif
#endif

#ifndef PURPLE_LCH
#ifdef CIE_D50
#define PURPLE_LCH vec3(31.04,30.66,316.69)
#else
#define PURPLE_LCH vec3(30.87,32.78,317.71)
#endif
#endif

#ifndef PURPLE_XYY
#define PURPLE_XYY vec3(0.285, 0.202, 6.6)
#endif

// Yellow green	(5 GY 7.1/9.1)
#ifndef YELLOW_GREEN
// sRGB: 159	190	64
#define YELLOW_GREEN vec3(0.624, 0.745, 0.227)
#endif

#ifndef YELLOW_GREEN_XYZ
#ifdef CIE_D50
#define YELLOW_GREEN_XYZ vec3(35.701,45.082,9.377)
#else
#define YELLOW_GREEN_XYZ vec3(33.992,44.969,11.835)
#endif
#endif

#ifndef YELLOW_GREEN_LAB
#ifdef CIE_D50
#define YELLOW_GREEN_LAB vec3(72.95,-24.35,56.48)
#else
#define YELLOW_GREEN_LAB vec3(72.87,-28.16,57.78)
#endif
#endif

#ifndef YELLOW_GREEN_LCH
#ifdef CIE_D50
#define YELLOW_GREEN_LCH vec3(72.95,61.51,113.32)
#else
#define YELLOW_GREEN_LCH vec3(72.87,64.28,115.98)
#endif
#endif

#ifndef YELLOW_GREEN_XYY
#define YELLOW_GREEN_XYY vec3(0.380, 0.489, 44.3)
#endif

// Orange Yellow	(10 YR 7/10.5)
#ifndef ORANGE_YELLOW
// sRGB: 228	162	41
#define ORANGE_YELLOW vec3(0.894, 0.635, 0.160)
#endif

#ifndef ORANGE_YELLOW_XYZ
#ifdef CIE_D50
#define ORANGE_YELLOW_XYZ vec3(49.37,44.438,6.215)
#else
#define ORANGE_YELLOW_XYZ vec3(45.74,42.941,8.056)
#endif
#endif

#ifndef ORANGE_YELLOW_LAB
#ifdef CIE_D50
#define ORANGE_YELLOW_LAB vec3(72.52,18.45,68.16)
#else
#define ORANGE_YELLOW_LAB vec3(71.51,14.6,66.93)
#endif
#endif

#ifndef ORANGE_YELLOW_LCH
#ifdef CIE_D50
#define ORANGE_YELLOW_LCH vec3(72.52,70.62,74.85)
#else
#define ORANGE_YELLOW_LCH vec3(71.51,68.5,77.69)
#endif
#endif

#ifndef ORANGE_YELLOW_XYY
#define ORANGE_YELLOW_XYY vec3(0.473, 0.438, 43.1)
#endif

// Blue	(7.5 PB 2.9/12.7)
#ifndef BLUE
// sRGB: 45	63	149
#define BLUE vec3(0.176, 0.247, 0.584)
#endif

#ifndef BLUE_XYZ
#ifdef CIE_D50
#define BLUE_XYZ vec3(7.305,6.022,22.182)
#else
#define BLUE_XYZ vec3(8.374,6.347,29.467)
#endif
#endif

#ifndef BLUE_LAB
#ifdef CIE_D50
#define BLUE_LAB vec3(29.47,15.59,-50.68)
#else
#define BLUE_LAB vec3(30.27,23.04,-49.59)
#endif
#endif

#ifndef BLUE_LCH
#ifdef CIE_D50
#define BLUE_LCH vec3(29.47,53.03,287.1)
#else
#define BLUE_LCH vec3(30.27,54.68,294.92)
#endif
#endif

#ifndef BLUE_XYY
#define BLUE_XYY vec3(0.187, 0.129, 6.1)
#endif

// Green (0.25 G 5.4/9.6)
#ifndef GREEN
// sRGB: 70	150	74
#define GREEN vec3(0.239, 0.588, 0.290)
#endif

#ifndef GREEN_XYZ
#ifdef CIE_D50
#define GREEN_XYZ vec3(15.229,23.542,8.222)
#else
#define GREEN_XYZ vec3(14.881,23.932,10.401)
#endif
#endif

#ifndef GREEN_LAB
#ifdef CIE_D50
#define GREEN_LAB vec3(55.63,-38.46,30.77)
#else
#define GREEN_LAB vec3(56.02,-40.94,32.75)
#endif
#endif

#ifndef GREEN_LCH
#ifdef CIE_D50
#define GREEN_LCH vec3(55.63,49.26,141.34)
#else
#define GREEN_LCH vec3(56.02,52.43,141.35)
#endif
#endif

#ifndef GREEN_XYY
#define GREEN_XYY vec3(0.305, 0.478, 23.4)
#endif

// Red (5 R 4/12)
#ifndef RED
// sRGB: 176	57	58
#define RED vec3(0.690, 0.224, 0.227)
#endif

#ifndef RED_XYZ
#ifdef CIE_D50
#define RED_XYZ vec3(22.59,13.543,4.08)
#else
#define RED_XYZ vec3(20.197,12.553,5.383)
#endif
#endif

#ifndef RED_LAB
#ifdef CIE_D50
#define RED_LAB vec3(43.57,51.47,29.3)
#else
#define RED_LAB vec3(42.08,48.02,26.74)
#endif
#endif

#ifndef RED_LCH
#ifdef CIE_D50
#define RED_LCH vec3(43.57,59.22,29.65)
#else
#define RED_LCH vec3(42.08,54.96,29.11)
#endif
#endif

#ifndef RED_XYY
#define RED_XYY vec3(0.539, 0.313, 12.0)
#endif

// Yellow (5 Y 8/11.1)
#ifndef YELLOW
// SRGB: 236	200	24
#define YELLOW vec3(0.925, 0.784, 0.094)
#endif

#ifndef YELLOW_XYZ
#ifdef CIE_D50
#define YELLOW_XYZ vec3(60.014,60.949,7.554)
#else
#define YELLOW_XYZ vec3(55.779,59.612,9.448)
#endif
#endif

#ifndef YELLOW_LAB
#ifdef CIE_D50
#define YELLOW_LAB vec3(82.35,2.97,79.44)
#else
#define YELLOW_LAB vec3(81.63,-2.19,79.78)
#endif
#endif

#ifndef YELLOW_LCH
#ifdef CIE_D50
#define YELLOW_LCH vec3(82.35,79.49,87.86)
#else
#define YELLOW_LCH vec3(81.63,79.81,91.57)
#endif
#endif

#ifndef YELLOW_XYY
#define YELLOW_XYY vec3(0.448, 0.470, 59.1)
#endif

// Magenta (2.5 RP 5/12)
#ifndef MAGENTA
// sRGB: 191	86	152
#define MAGENTA vec3(0.749, 0.309, 0.598)
#endif

#ifndef MAGENTA_XYZ
#ifdef CIE_D50
#define MAGENTA_XYZ vec3(32.305,20.971,23.891)
#else
#define MAGENTA_XYZ vec3(30.689,20.117,32.074)
#endif
#endif

#ifndef MAGENTA_LAB
#ifdef CIE_D50
#define MAGENTA_LAB vec3(52.92,50.21,-13.48)
#else
#define MAGENTA_LAB vec3(51.97,50.05,-15.89)
#endif
#endif

#ifndef MAGENTA_LCH
#ifdef CIE_D50
#define MAGENTA_LCH vec3(52.92,51.99,344.97)
#else
#define MAGENTA_LCH vec3(51.97,52.51,342.39)
#endif
#endif

#ifndef MAGENTA_XYY
#define MAGENTA_XYY vec3(0.364, 0.233, 19.8)
#endif

// Cyan (5 B 5/8)
#ifndef CYAN
// sRGB: 0	137	168
#define CYAN vec3(0.000, 0.537, 0.659)
#endif

#ifndef CYAN_XYZ
#ifdef CIE_D50
#define CYAN_XYZ vec3(13.964,19.428,31.039)
#else
#define CYAN_XYZ vec3(15.131,20.357,40.473)
#endif
#endif

#ifndef CYAN_LAB
#ifdef CIE_D50
#define CYAN_LAB vec3(51.18,-27.02,-28.54)
#else
#define CYAN_LAB vec3(52.24,-23.14,-26.15)
#endif
#endif

#ifndef CYAN_LCH
#ifdef CIE_D50
#define CYAN_LCH vec3(51.18,39.29,226.57)
#else
#define CYAN_LCH vec3(52.24,34.92,228.49)
#endif
#endif

#ifndef CYAN_XYY
#define CYAN_XYY vec3(0.196, 0.252, 19.8)
#endif

// White (N 9.5/)
#ifndef WHITE
// sRGB: 244	244	241
#define WHITE vec3(0.956, 0.956, 0.945)
#endif

#ifndef WHITE_XYZ
#ifdef CIE_D50
#define WHITE_XYZ vec3(87.473,90.892,73.275)
#else
#define WHITE_XYZ vec3(86.047,90.868,96.433)
#endif
#endif

#ifndef WHITE_LAB
#ifdef CIE_D50
#define WHITE_LAB vec3(96.37,-0.31,1.5)
#else
#define WHITE_LAB vec3(96.36,-0.6,1.65)
#endif
#endif

#ifndef WHITE_LCH
#ifdef CIE_D50
#define WHITE_LCH vec3(96.37,1.53,101.57)
#else
#define WHITE_LCH vec3(96.36,1.76,109.99)
#endif
#endif

#ifndef WHITE_XYY
#define WHITE_XYY vec3(0.310, 0.316, 90.0)
#endif

// Neutral 8 (N 8/)
#ifndef NEUTRAL_80
// sRGB: 201	203	203
#define NEUTRAL_80 vec3(0.789, 0.797, 0.797)
#endif

#ifndef NEUTRAL_80_XYZ
#ifdef CIE_D50
#define NEUTRAL_80_XYZ vec3(57.342,59.788,49.481)
#else
#define NEUTRAL_80_XYZ vec3(56.562,59.821,65.218)
#endif
#endif

#ifndef NEUTRAL_80_LAB
#ifdef CIE_D50
#define NEUTRAL_80_LAB vec3(81.72,-0.75,-0.16)
#else
#define NEUTRAL_80_LAB vec3(81.74,-0.73,-0.07)
#endif
#endif

#ifndef NEUTRAL_80_LCH
#ifdef CIE_D50
#define NEUTRAL_80_LCH vec3(81.72,0.77,192.3)
#else
#define NEUTRAL_80_LCH vec3(81.74,0.74,185.65)
#endif
#endif

#ifndef NEUTRAL_80_XYY
#define NEUTRAL_80_XYY vec3(0.310, 0.316, 59.1)
#endif

// Neutral 6.5	(N 6.5/)
#ifndef NEUTRAL_65
// sRGB: 162	164	164
#define NEUTRAL_65 vec3(0.635, 0.643, 0.643)
#endif

#ifndef NEUTRAL_65_XYZ
#ifdef CIE_D50
#define NEUTRAL_65_XYZ vec3(35.589,37.181,30.911)
#else
#define NEUTRAL_65_XYZ vec3(35.144,37.209,40.764)
#endif
#endif

#ifndef NEUTRAL_65_LAB
#ifdef CIE_D50
#define NEUTRAL_65_LAB vec3(67.41,-0.88,-0.36)
#else
#define NEUTRAL_65_LAB vec3(67.43,-0.75,-0.29)
#endif
#endif

#ifndef NEUTRAL_65_LCH
#ifdef CIE_D50
#define NEUTRAL_65_LCH vec3(67.41,0.95,202.11)
#else
#define NEUTRAL_65_LCH vec3(67.43,0.81,201.35)
#endif
#endif

#ifndef NEUTRAL_65_XYY
#define NEUTRAL_65_XYY vec3(0.310, 0.316, 36.2)
#endif

// Neutral 5 (N 5/)
#ifndef NEUTRAL_50
// sRGB: 121	122	122
#define NEUTRAL_50 vec3(0.475, 0.478, 0.478)
#endif

#ifndef NEUTRAL_50_XYZ
#ifdef CIE_D50
#define NEUTRAL_50_XYZ vec3(18.752,19.493,16.152)
#else
#define NEUTRAL_50_XYZ vec3(18.505,19.497,21.306)
#endif
#endif

#ifndef NEUTRAL_50_LAB
#ifdef CIE_D50
#define NEUTRAL_50_LAB vec3(51.26,-0.22,-0.16)
#else
#define NEUTRAL_50_LAB vec3(51.26,-0.14,-0.14)
#endif
#endif

#ifndef NEUTRAL_50_LCH
#ifdef CIE_D50
#define NEUTRAL_50_LCH vec3(51.26,0.28,215.6)
#else
#define NEUTRAL_50_LCH vec3(51.26,0.2,225.42)
#endif
#endif

#ifndef NEUTRAL_50_XYY
#define NEUTRAL_50_XYY vec3(0.310, 0.316, 19.8)
#endif


// Neutral 3.5	(N 3.5/)
#ifndef NEUTRAL_35
// sRGB: 84	85	86
#define NEUTRAL_35  vec3(0.329, 0.333, 0.337)
#endif

#ifndef NEUTRAL_35_XYZ
#ifdef CIE_D50
#define NEUTRAL_35_XYZ vec3(8.833,9.223,7.82)
#else
#define NEUTRAL_35_XYZ vec3(8.737,9.233,10.323)
#endif 
#endif

#ifndef NEUTRAL_35_LAB
#ifdef CIE_D50
#define NEUTRAL_35_LAB vec3(36.41,-0.5,-0.82)
#else
#define NEUTRAL_35_LAB vec3(36.43,-0.33,-0.8)
#endif
#endif

#ifndef NEUTRAL_35_LCH
#ifdef CIE_D50
#define NEUTRAL_35_LCH vec3(36.41,0.96,238.548)
#else
#define NEUTRAL_35_LCH vec3(36.43,0.87,247.43)
#endif
#endif

#ifndef NEUTRAL_35_XYY
#define NEUTRAL_35_XYY vec3(0.310, 0.316, 9.0)
#endif

// Black (N 2/)
#ifndef BLACK
// sRGB: 51	51	52
#define BLACK vec3(0.200, 0.200, 0.204)
#endif

#ifndef BLACK_XYZ
#ifdef CIE_D50
#define BLACK_XYZ vec3(3.225,3.34,2.822)
#else
#define BLACK_XYZ vec3(3.185,3.342,3.727)
#endif
#endif

#ifndef BLACK_LAB
#ifdef CIE_D50
#define BLACK_LAB vec3(21.36,0.07,-0.5)
#else
#define BLACK_LAB vec3(21.36,0.14,-0.51)
#endif
#endif

#ifndef BLACK_LCH
#ifdef CIE_D50
#define BLACK_LCH vec3(21.36,0.51,278.09)
#else
#define BLACK_LCH vec3(21.36,0.53,285.38)
#endif
#endif

#ifndef BLACK_XYY
#define BLACK_XYY vec3(0.310, 0.316, 3.1)
#endif

#ifndef FNC_PALETTE_MACBETH
#define FNC_PALETTE_MACBETH
vec3 macbeth (const int index) {
    vec3 rgb[24];
    rgb[0] = DARK_SKIN;
    rgb[1] = LIGHT_SKIN;
    rgb[2] = BLUE_SKY;
    rgb[3] = FOLIAGE;
    rgb[4] = BLUE_FLOWER;
    rgb[5] = BLUISH_GREEN;
    rgb[6] = ORANGE;
    rgb[7] = PURPLISH_BLUE;
    rgb[8] = MODERATE_RED;
    rgb[9] = PURPLE;
    rgb[10] = YELLOW_GREEN;
    rgb[11] = ORANGE_YELLOW;
    rgb[12] = BLUE;
    rgb[13] = GREEN;
    rgb[14] = RED;
    rgb[15] = YELLOW;
    rgb[16] = MAGENTA;
    rgb[17] = CYAN;
    rgb[18] = WHITE;
    rgb[19] = NEUTRAL_80;
    rgb[20] = NEUTRAL_65;
    rgb[21] = NEUTRAL_50;
    rgb[22] = NEUTRAL_35;
    rgb[23] = BLACK;

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 64; i++)
        if (i == index) return rgb[i];
    #else
    return rgb[index];
    #endif
}

vec3 macbethXYZ (const int index) {
    vec3 xyz[24];
    xyz[0] = DARK_SKIN_XYZ;
    xyz[1] = LIGHT_SKIN_XYZ;
    xyz[2] = BLUE_SKY_XYZ;
    xyz[3] = FOLIAGE_XYZ;
    xyz[4] = BLUE_FLOWER_XYZ;
    xyz[5] = BLUISH_GREEN_XYZ;
    xyz[6] = ORANGE_XYZ;
    xyz[7] = PURPLISH_BLUE_XYZ;
    xyz[8] = MODERATE_RED_XYZ;
    xyz[9] = PURPLE_XYZ;
    xyz[10] = YELLOW_GREEN_XYZ;
    xyz[11] = ORANGE_YELLOW_XYZ;
    xyz[12] = BLUE_XYZ;
    xyz[13] = GREEN_XYZ;
    xyz[14] = RED_XYZ;
    xyz[15] = YELLOW_XYZ;
    xyz[16] = MAGENTA_XYZ;
    xyz[17] = CYAN_XYZ;
    xyz[18] = WHITE_XYZ;
    xyz[19] = NEUTRAL_80_XYZ;
    xyz[20] = NEUTRAL_65_XYZ;
    xyz[21] = NEUTRAL_50_XYZ;
    xyz[22] = NEUTRAL_35_XYZ;
    xyz[23] = BLACK_XYZ;
    
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 24; i++)
        if (i == index) return xyz[i];
    #else
    return xyz[index];
    #endif
}

vec3 macbethLAB (const int index) {
    vec3 lab[24];
    lab[0] = DARK_SKIN_LAB;
    lab[1] = LIGHT_SKIN_LAB;
    lab[2] = BLUE_SKY_LAB;
    lab[3] = FOLIAGE_LAB;
    lab[4] = BLUE_FLOWER_LAB;
    lab[5] = BLUISH_GREEN_LAB;
    lab[6] = ORANGE_LAB;
    lab[7] = PURPLISH_BLUE_LAB;
    lab[8] = MODERATE_RED_LAB;
    lab[9] = PURPLE_LAB;
    lab[10] = YELLOW_GREEN_LAB;
    lab[11] = ORANGE_YELLOW_LAB;
    lab[12] = BLUE_LAB;
    lab[13] = GREEN_LAB;
    lab[14] = RED_LAB;
    lab[15] = YELLOW_LAB;
    lab[16] = MAGENTA_LAB;
    lab[17] = CYAN_LAB;
    lab[18] = WHITE_LAB;
    lab[19] = NEUTRAL_80_LAB;
    lab[20] = NEUTRAL_65_LAB;
    lab[21] = NEUTRAL_50_LAB;
    lab[22] = NEUTRAL_35_LAB;
    lab[23] = BLACK_LAB;

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 24; i++)
        if (i == index) return lab[i];
    #else
    return lab[index];
    #endif
}

vec3 macbethLCH (const int index) {
    vec3 lch[24];
    lch[0] = DARK_SKIN_LCH;
    lch[1] = LIGHT_SKIN_LCH;
    lch[2] = BLUE_SKY_LCH;
    lch[3] = FOLIAGE_LCH;
    lch[4] = BLUE_FLOWER_LCH;
    lch[5] = BLUISH_GREEN_LCH;
    lch[6] = ORANGE_LCH;
    lch[7] = PURPLISH_BLUE_LCH;
    lch[8] = MODERATE_RED_LCH;
    lch[9] = PURPLE_LCH;
    lch[10] = YELLOW_GREEN_LCH;
    lch[11] = ORANGE_YELLOW_LCH;
    lch[12] = BLUE_LCH;
    lch[13] = GREEN_LCH;
    lch[14] = RED_LCH;
    lch[15] = YELLOW_LCH;
    lch[16] = MAGENTA_LCH;
    lch[17] = CYAN_LCH;
    lch[18] = WHITE_LCH;
    lch[19] = NEUTRAL_80_LCH;
    lch[20] = NEUTRAL_65_LCH;
    lch[21] = NEUTRAL_50_LCH;
    lch[22] = NEUTRAL_35_LCH;
    lch[23] = BLACK_LCH;
    
    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 24; i++)
        if (i == index) return lch[i];
    #else
    return lch[index];
    #endif
}

vec3 macbethXYY (const int index) {
    vec3 xyy[24];
    xyy[0] = DARK_SKIN_XYY;
    xyy[1] = LIGHT_SKIN_XYY;
    xyy[2] = BLUE_SKY_XYY;
    xyy[3] = FOLIAGE_XYY;
    xyy[4] = BLUE_FLOWER_XYY;
    xyy[5] = BLUISH_GREEN_XYY;
    xyy[6] = ORANGE_XYY;
    xyy[7] = PURPLISH_BLUE_XYY;
    xyy[8] = MODERATE_RED_XYY;
    xyy[9] = PURPLE_XYY;
    xyy[10] = YELLOW_GREEN_XYY;
    xyy[11] = ORANGE_YELLOW_XYY;
    xyy[12] = BLUE_XYY;
    xyy[13] = GREEN_XYY;
    xyy[14] = RED_XYY;
    xyy[15] = YELLOW_XYY;
    xyy[16] = MAGENTA_XYY;
    xyy[17] = CYAN_XYY;
    xyy[18] = WHITE_XYY;
    xyy[19] = NEUTRAL_80_XYY;
    xyy[20] = NEUTRAL_65_XYY;
    xyy[21] = NEUTRAL_50_XYY;
    xyy[22] = NEUTRAL_35_XYY;
    xyy[23] = BLACK_XYY;

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < 24; i++)
        if (i == index) return xyy[i];
    #else
    return xyy[index];
    #endif
}

#endif