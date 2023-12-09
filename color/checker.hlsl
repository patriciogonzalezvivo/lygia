
// MACBETH COLOR PALETTE
//
// Values from:
//  - http://en.wikipedia.org/wiki/ColorChecker
//  - http://kurtmunger.com/color_checkerid277.html

//Dark skin		3 YR 3.7/3.2	0.400 0.350 10.1	#735244
#ifndef DARK_SKIN
#define DARK_SKIN float3(0.454, 0.311, 0.255)
#endif

//Light skin	2.2 YR 6.47/4.1	0.377 0.345 35.8	#c29682
#ifndef LIGHT_SKIN
#define LIGHT_SKIN float3(0.773, 0.563, 0.497)
#endif

//Blue sky		4.3 PB 4.95/5.5	0.247 0.251 19.3	#627a9d
#ifndef BLUE_SKY
#define BLUE_SKY float3(0.356, 0.472, 0.609)
#endif

//Foliage		6.7 GY 4.2/4.1	0.337 0.422 13.3	#576c43
#ifndef FOLIAGE
#define FOLIAGE float3(0.359, 0.425, 0.250)
#endif

//Blue flower	9.7 PB 5.47/6.7	0.265 0.240 24.3	#8580b1
#ifndef BLUE_FLOWER
#define BLUE_FLOWER float3(0.514, 0.498, 0.685)
#endif

//Bluish green	2.5 BG 7/6		0.261 0.343 43.1	#67bdaa
#ifndef BLUISH_GREEN
#define BLUISH_GREEN float3(0.374, 0.740, 0.673)
#endif

//Orange		5 YR 6/11		0.506 0.407 30.1	#d67e2c
#ifndef ORANGE
#define ORANGE float3(0.879, 0.487, 0.189)
#endif

//Purplish blue	7.5 PB 4/10.7	0.211 0.175 12.0	#505ba6
#ifndef PURPLISH_BLUE
#define PURPLISH_BLUE float3(0.269, 0.351, 0.656)
#endif

//Moderate red	2.5 R 5/10		0.453 0.306 19.8	#c15a63
#ifndef MODERATE_RED
#define MODERATE_RED float3(0.774, 0.313, 0.373)
#endif

//Purple		5 P 3/7			0.285 0.202 6.6		#5e3c6c
#ifndef PURPLE
#define PURPLE float3(0.364, 0.227, 0.407)
#endif

//Yellow green	5 GY 7.1/9.1	0.380 0.489 44.3	#9dbc40
#ifndef YELLOW_GREEN
#define YELLOW_GREEN float3(0.612, 0.734, 0.228)
#endif

//Orange Yellow	10 YR 7/10.5	0.473 0.438 43.1	#e0a32e
#ifndef ORANGE_YELLOW
#define ORANGE_YELLOW float3(0.892, 0.633, 0.153)
#endif

//Blue			7.5 PB 2.9/12.7	0.187 0.129 6.1		#383d96
#ifndef BLUE
#define BLUE float3(0.156, 0.241, 0.570)
#endif

//Green			0.25 G 5.4/9.6	0.305 0.478 23.4	#469449
#ifndef GREEN
#define GREEN float3(0.238, 0.577, 0.275)
#endif

//Red			5 R 4/12		0.539 0.313 12.0	#af363c
#ifndef RED
#define RED float3(0.699, 0.210, 0.222)
#endif

//Yellow		5 Y 8/11.1		0.448 0.470 59.1	#e7c71f
#ifndef YELLOW
#define YELLOW float3(0.927, 0.782, 0.057)
#endif

//Magenta		2.5 RP 5/12		0.364 0.233 19.8	#bb5695
#ifndef MAGENTA
#define MAGENTA float3(0.750, 0.309, 0.574)
#endif

//Cyan			5 B 5/8			0.196 0.252 19.8	#0885a1
#ifndef CYAN
#define CYAN float3(0.000, 0.521, 0.648)
#endif

//White			N 9.5/			0.310 0.316 90.0	#f3f3f2
#ifndef WHITE
#define WHITE float3(0.945, 0.948, 0.923)
#endif

//Neutral 8		N 8/			0.310 0.316 59.1	#c8c8c8
#ifndef NEUTRAL_80
#define NEUTRAL_80 float3(0.789, 0.793, 0.788)
#endif

//Neutral 6.5	N 6.5/			0.310 0.316 36.2	#a0a0a0
#ifndef NEUTRAL_65
#define NEUTRAL_65 float3(0.632, 0.640, 0.638)
#endif

//Neutral 5		N 5/			0.310 0.316 19.8	#7a7a79
#ifndef NEUTRAL_50
#define NEUTRAL_50 float3(0.473, 0.474, 0.473)
#endif

//Neutral 3.5	N 3.5/			0.310 0.316 9.0		#555555
#ifndef NEUTRAL_35
#define NEUTRAL_35 float3(0.324, 0.330, 0.331)
#endif

//Black			N 2/			0.310 0.316 3.1		#343434
#ifndef BLACK
#define BLACK float3(0.194, 0.195, 0.197)
#endif

// 48 SPYDERCHECKR COLOR PALETTE
// 
// Values from:
//  - https://www.northlight-images.co.uk/datacolor-spydercheckr-colour-test-card-review/
//  - https://www.bartneck.de/2017/10/24/patch-color-definitions-for-datacolor-spydercheckr-48/

// A1
// Lab: 61.35,  34.81,  18.38
// 0.071, 0.107
// sRGB: 210, 121, 117
#ifndef LOW_SAT_RED
#define LOW_SAT_RED float3(0.824, 0.475, 0.459)
#endif

// A2
// Lab: 75.50 ,  5.84,  50.42
// 0.071, 0.264
// sRGB: 216 	179 	90
#ifndef LOW_SAT_YELLOW
#define LOW_SAT_YELLOW float3(0.847, 0.702, 0.353)
#endif

// A3
// Lab: 66.82,	-25.1,	23.47
// 0.071, 0.421
// sRGB: 127 	175 	120
#ifndef LOW_SAT_GREEN
#define LOW_SAT_GREEN float3(0.498, 0.686, 0.471)
#endif

// A4
// Lab: 60.53,	-22.6, -20.40 
// 0.071, 0.579
// sRGB: 66 	157 	179
#ifndef LOW_SAT_CYAN
#define LOW_SAT_CYAN float3(0.259, 0.616, 0.702)
#endif

// A5
// Lab: 59.66,	-2.03, -28.46 
// 0.071, 0.736
// sRGB: 116 	147 	194
#ifndef LOW_SAT_BLUE
#define LOW_SAT_BLUE float3(0.455, 0.576, 0.761)
#endif

// A6
// Lab: 59.15,	30.83,  -5.72 
// 0.071, 0.893
// sRGB: 190 	121 	154
#ifndef LOW_SAT_MAGENTA
#define LOW_SAT_MAGENTA float3(0.745, 0.475, 0.604)
#endif

// B1
// Lab: 82.68,	 5.03,	 3.02
// 0.175, 0.107
// sRGB: 218 	203 	201
#ifndef RED_TINT_10
#define RED_TINT_10 float3(0.855, 0.796, 0.788)
#endif

// B2
// Lab: 82.25,	-2.42,	 3.78
// 0.175, 0.264
// sRGB: 203 	205 	196
#ifndef GREEN_TINT_10
#define GREEN_TINT_10 float3(0.796, 0.804, 0.769)
#endif

// B3
// Lab: 82.29,	 2.20,	-2.04
// 0.175, 0.421
// sRGB: 206 	203 	208
#ifndef BLUE_TINT_10
#define BLUE_TINT_10 float3(0.808, 0.796, 0.816)
#endif

// B4
// Lab: 24.89,	 4.43,	 0.78
// 0.175, 0.579
// sRGB: 66 	57 	58
#ifndef RED_TONE_90
#define RED_TONE_90 float3(0.259, 0.224, 0.227)
#endif

// B5
// Lab: 25.16,	-3.88,	 2.13
// 0.175, 0.736
// sRGB: 54 	61 	56
#ifndef GREEN_TONE_90
#define GREEN_TONE_90 float3(0.212, 0.239, 0.220)
#endif

// B6
// Lab: 26.13,	 2.61,	-5.03
// 0.175, 0.893
// sRGB: 63 	60 	69
#ifndef BLUE_TONE_90
#define BLUE_TONE_90 float3(0.247, 0.235, 0.271)
#endif

// C1
// Lab: 85.42,	 9.41,	14.49
// 0.279, 0.107
// sRGB: 237 	206 	186
#ifndef LIGHTEST_SKIN
#define LIGHTEST_SKIN float3(0.929, 0.808, 0.729)
#endif

// C2
// Lab: 74.28,	 9.05,	27.21
// 0.279, 0.264
// sRGB: 211 	175 	133
#ifndef LIGHTER_SKIN
#define LIGHTER_SKIN float3(0.827, 0.686, 0.522)
#endif

// C3
// Lab: 64.57,	12.39,	37.24
// 0.279, 0.421
// sRGB: 193 	149 	91
#ifndef MODERATE_SKIN
#define MODERATE_SKIN float3(0.757, 0.584, 0.357)
#endif

// C4
// Lab: 44.49,	17.23,	26.24
// 0.279, 0.579
// sRGB: 139 	93 	61
#ifndef MEDIUM_SKIN
#define MEDIUM_SKIN float3(0.545, 0.365, 0.239)
#endif

// C5
// Lab: 25.29,	 7.95,	 8.87
// 0.279, 0.736
// sRGB: 74 	55 	46
#ifndef DEEP_SKIN
#define DEEP_SKIN float3(0.290, 0.216, 0.180)
#endif

// C6
// Lab: 22.67,	 2.11,	-1.10
// 0.279, 0.893
// sRGB: 57 	54 	56
#ifndef GRAY_95
#define GRAY_95 float3(0.224, 0.212, 0.220)
#endif

// D1
// Lab: 92.72,	 1.89,	 2.76
// 0.384, 0.107
// sRGB: 241 	233 	229
#ifndef GRAY_05
#define GRAY_05 float3(0.945, 0.914, 0.898)
#endif

// D2
// Lab: 88.85,	 1.59,	 2.27
// 0.384, 0.264
// sRGB: 229 	222 	220
#ifndef GRAY_10
#define GRAY_10 float3(0.898, 0.871, 0.863)
#endif

// D3
// Lab: 73.42,	 0.99,	 1.89
// 0.384, 0.421
//sRGB 182 	178 	176
#ifndef GRAY_30
#define GRAY_30 float3(0.714, 0.698, 0.690)
#endif

// D4
// Lab: 57.15,	 0.57,	 1.19
// 0.384, 0.579
// sRGB: 139 	136 	135
#ifndef GRAY_50
#define GRAY_50 float3(0.545, 0.533, 0.529)
#endif

// D5
// Lab: 41.57,	 0.24,	 1.45
// 0.384, 0.736
// sRGB: 100 	99 	97
#ifndef GRAY_70
#define GRAY_70 float3(0.392, 0.388, 0.380)
#endif

// D6
// Lab: 25.65,	 1.24,	 0.05
// 0.384, 0.893
// sRGB: 63 	61 	62
#ifndef GRAY_90
#define GRAY_90 float3(0.247, 0.239, 0.243)
#endif

// E1
// Lab: 96.04,	 2.16,	 2.60 
// 0.616, 0.107
// sRGB: 249, 242, 238
#ifndef CARD_WHITE
#define CARD_WHITE float3(0.976, 0.949, 0.933)
#endif

// E2
// Lab: 80.44,	 1.17,	 2.05 
// 0.616, 0.264
// sRGB: 202, 198, 195
#ifndef GRAY_20
#define GRAY_20 float3(0.792, 0.777, 0.765)
#endif

// E3
// Lab: 65.52,	 0.69,	 1.86 
// 0.616, 0.421
// sRGB: 161, 157, 154
#ifndef GRAY_40
#define GRAY_40 float3(0.631, 0.616, 0.604)
#endif

// E4
// Lab: 49.62,	 0.58,	 1.56 
// 0.616, 0.579
// sRGB: 122, 118, 116
#ifndef GRAY_60
#define GRAY_60 float3(0.478, 0.463, 0.455)
#endif

// E5
// Lab: 33.55,	 0.35,	 1.40 
// 0.616, 0.736
// sRGB: 80, 80, 78
#ifndef GRAY_80
#define GRAY_80 float3(0.314, 0.314, 0.306)
#endif

// E6
// Lab: 16.91,	 1.43,	-0.81 
// 0.616, 0.893
// sRGB: 43, 41, 43
#ifndef CARD_BLACK
#define CARD_BLACK float3(0.169, 0.161, 0.169)
#endif

// F1
// Lab: 47.12, -32.50, -28.75
// 0.721, 0.107
// sRGB: 0, 127, 159
#ifndef PRIMARY_CYAN
#define PRIMARY_CYAN float3(0.000, 0.498, 0.623)
#endif

// F2
// Lab: 50.49,	53.45, -13.55 
//  0.721, 0.264
// sRGB: 192, 75, 145
#ifndef PRIMARY_MAGENTA
#define PRIMARY_MAGENTA float3(0.753, 0.294, 0.569)
#endif

// F3
// Lab: 83.61,	 3.36,	87.02
// 0.721, 0.421
// sRGB: 245, 205, 0
#ifndef PRIMARY_YELLOW
#define PRIMARY_YELLOW float3(0.961, 0.804, 0.000)
#endif 

// F4
// Lab: 41.05,	60.75,	31.17
// 0.721, 0.579
// sRGB: 186, 26, 51
#ifndef PRIMARY_RED
#define PRIMARY_RED float3(0.729, 0.102, 0.200)
#endif

// F5
// Lab: 54.14, -40.80,	34.75 
// 0.721, 0.736
// sRGB: 57, 146, 64
#ifndef PRIMARY_GREEN
#define PRIMARY_GREEN float3(0.224, 0.573, 0.251)
#endif

// F6
// Lab: 24.75,	13.78, -49.48 
// 0.721, 0.893
// sRGB: 25, 55, 135
#ifndef PRIMARY_BLUE
#define PRIMARY_BLUE float3(0.098, 0.216, 0.529)
#endif

// G1
// Lab: 60.94,	38.21,	61.31 
// 0.825, 0.107
// sRGB: 222, 118, 32
#ifndef PRIMARY_ORANGE
#define PRIMARY_ORANGE float3(0.871, 0.463, 0.125)
#endif

// G2
// Lab: 37.80,	 7.30, -43.04
// 0.825, 0.26
// sRGB: 58, 89, 160
#ifndef BLUEPRINT
#define BLUEPRINT float3(0.227, 0.349, 0.627)
#endif

// G3
// Lab: 49.81,	48.50,	15.76 
// 0.825, 0.421
// sRGB: 195, 79, 95
#ifndef PINK
#define PINK float3(0.765, 0.310, 0.373)
#endif

// G4
// Lab: 28.88,	19.36, -24.48
// 0.825, 0.57
// sRGB: 83, 58, 106
#ifndef VIOLET
#define VIOLET float3(0.325, 0.227, 0.416)
#endif

// G5
// Lab: 72.45, -23.60,	60.47
// 0.825, 0.73
// sRGB: 157, 188, 54
#ifndef APPLE_GREEN
#define APPLE_GREEN float3(0.616, 0.737, 0.212)
#endif

// G6
// Lab: 71.65,	23.74,	72.28 
// 0.825, 0.893
// sRGB: 238, 158, 25
#ifndef SUNFLOWER
#define SUNFLOWER float3(0.933, 0.620, 0.098)
#endif

// H1
// Lab: 70.19, -31.90,	 1.98
// 0.929, 0.107
// sRGB: 98, 187, 166
#ifndef AQUA
#define AQUA float3(0.384, 0.733, 0.651)
#endif

// H2
// Lab: 54.38,	 8.84, -25.71
// 0.929, 0.264
// sRGB: 126, 125, 174
#ifndef LAVANDER
#define LAVANDER float3(0.494, 0.490, 0.682)
#endif

// H3
// Lab: 42.03, -15.80,	22.93
// 0.929, 0.421
// sRGB: 82, 106, 60
#ifndef EVERGREEN
#define EVERGREEN float3(0.322, 0.423, 0.247)
#endif

// H4
// Lab: 48.82,	-5.11, -23.08
// 0.929, 0.579
// sRGB: 87, 120, 155
#ifndef STEEL_BLUE
#define STEEL_BLUE float3(0.341, 0.467, 0.603)
#endif

// H5
// Lab: 65.10,	18.14,	18.68 
// 0.929, 0.736 
// sRGB: 197, 145, 125
#ifndef CLASSIC_LIGHT_SKIN
#define CLASSIC_LIGHT_SKIN float3(0.769, 0.557, 0.494)
#endif

// H6
// Lab: 36.13,	14.15,	15.78 
// 0.929, 0.893 
// sRGB: 112, 76, 60
#ifndef CLASSIC_DARK_SKIN
#define CLASSIC_DARK_SKIN float3(0.439, 0.302, 0.247)
#endif