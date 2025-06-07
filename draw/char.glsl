#include "../math/modi.glsl"

/*
contributors: ['Tim Gfrerer', 'Patricio Gonzalez Vivo']
description: |
    Draws a character from a bitmap font. 
    Based on Tim's article about Texture-less Text Rendering https://poniesandlight.co.uk/reflect/debug_print_text/
use: <vec2> char(<vec2> uv, <int> char_code)
*/

#ifndef CHAR_SIZE
#define CHAR_SIZE vec2(.02)
#endif

#ifndef CHAR_TOTAL
#define CHAR_TOTAL 96
#endif

#define CHAR_SPACE 0
#define CHAR_EXCLAMATION 1
#define CHAR_QUOTE 2
#define CHAR_NUMBER 3
#define CHAR_DOLLAR 4
#define CHAR_PERCENT 5
#define CHAR_AMPERSAND 6
#define CHAR_APOSTROPHE 7
#define CHAR_PAREN_LEFT 8
#define CHAR_PAREN_RIGHT 9
#define CHAR_ASTERISK 10
#define CHAR_PLUS 11
#define CHAR_COMMA 12
#define CHAR_MINUS 13
#define CHAR_PERIOD 14
#define CHAR_SLASH 15
#define CHAR_0 16
#define CHAR_1 17
#define CHAR_2 18
#define CHAR_3 19
#define CHAR_4 20
#define CHAR_5 21
#define CHAR_6 22
#define CHAR_7 23
#define CHAR_8 24
#define CHAR_9 25
#define CHAR_COLON 26
#define CHAR_SEMICOLON 27
#define CHAR_LESS 28
#define CHAR_EQUAL 29
#define CHAR_GREATER 30
#define CHAR_QUESTION 31
#define CHAR_AT 32
#define CHAR_A 33
#define CHAR_B 34
#define CHAR_C 35
#define CHAR_D 36
#define CHAR_E 37
#define CHAR_F 38
#define CHAR_G 39
#define CHAR_H 40
#define CHAR_I 41
#define CHAR_J 42
#define CHAR_K 43
#define CHAR_L 44
#define CHAR_M 45
#define CHAR_N 46
#define CHAR_O 47
#define CHAR_P 48
#define CHAR_Q 49
#define CHAR_R 50
#define CHAR_S 51
#define CHAR_T 52
#define CHAR_U 53
#define CHAR_V 54
#define CHAR_W 55
#define CHAR_X 56
#define CHAR_Y 57
#define CHAR_Z 58
#define CHAR_BRACKET_LEFT 59
#define CHAR_BACKSLASH 60
#define CHAR_BRACKET_RIGHT 61
#define CHAR_CARET 62
#define CHAR_UNDERSCORE 63
#define CHAR_GRAVE 64
#define CHAR_a 65
#define CHAR_b 66
#define CHAR_c 67
#define CHAR_d 68
#define CHAR_e 69
#define CHAR_f 70
#define CHAR_g 71
#define CHAR_h 72
#define CHAR_i 73
#define CHAR_j 74
#define CHAR_k 75
#define CHAR_l 76
#define CHAR_m 77
#define CHAR_n 78
#define CHAR_o 79
#define CHAR_p 80
#define CHAR_q 81
#define CHAR_r 82
#define CHAR_s 83
#define CHAR_t 84
#define CHAR_u 85
#define CHAR_v 86
#define CHAR_w 87
#define CHAR_x 88
#define CHAR_y 89
#define CHAR_z 90
#define CHAR_BRACE_LEFT 91
#define CHAR_BAR 92
#define CHAR_BRACE_RIGHT 93
#define CHAR_TILDE 94

#ifndef FNC_CHAR
#define FNC_CHAR
ivec4 charLUT( const int index ) {
    ivec4 d[CHAR_TOTAL];
    d[0] = ivec4(0x0, 0x0, 0x0, 0x0);
    d[1] = ivec4(0x1010, 0x10101010, 0x1010, 0x0);
    d[2] = ivec4(0x242424, 0x24000000, 0x0, 0x0);
    d[3] = ivec4(0x24, 0x247e2424, 0x247e2424, 0x0);
    d[4] = ivec4(0x808, 0x1e20201c, 0x2023c08, 0x8000000);
    d[5] = ivec4(0x30, 0x494a3408, 0x16294906, 0x0);
    d[6] = ivec4(0x3048, 0x48483031, 0x49464639, 0x0);
    d[7] = ivec4(0x101010, 0x10000000, 0x0, 0x0);
    d[8] = ivec4(0x408, 0x8101010, 0x10101008, 0x8040000);
    d[9] = ivec4(0x2010, 0x10080808, 0x8080810, 0x10200000);
    d[10] = ivec4(0x0, 0x24187e, 0x18240000, 0x0);
    d[11] = ivec4(0x0, 0x808087f, 0x8080800, 0x0);
    d[12] = ivec4(0x0, 0x0, 0x1818, 0x8081000);
    d[13] = ivec4(0x0, 0x7e, 0x0, 0x0);
    d[14] = ivec4(0x0, 0x0, 0x1818, 0x0);
    d[15] = ivec4(0x202, 0x4040808, 0x10102020, 0x40400000);
    d[16] = ivec4(0x3c, 0x42464a52, 0x6242423c, 0x0);
    d[17] = ivec4(0x8, 0x18280808, 0x808083e, 0x0);
    d[18] = ivec4(0x3c, 0x42020204, 0x810207e, 0x0);
    d[19] = ivec4(0x7e, 0x4081c02, 0x202423c, 0x0);
    d[20] = ivec4(0x4, 0xc142444, 0x7e040404, 0x0);
    d[21] = ivec4(0x7e, 0x40407c02, 0x202423c, 0x0);
    d[22] = ivec4(0x1c, 0x2040407c, 0x4242423c, 0x0);
    d[23] = ivec4(0x7e, 0x2040408, 0x8101010, 0x0);
    d[24] = ivec4(0x3c, 0x4242423c, 0x4242423c, 0x0);
    d[25] = ivec4(0x3c, 0x4242423e, 0x2020438, 0x0);
    d[26] = ivec4(0x0, 0x181800, 0x1818, 0x0);
    d[27] = ivec4(0x0, 0x181800, 0x1818, 0x8081000);
    d[28] = ivec4(0x4, 0x8102040, 0x20100804, 0x0);
    d[29] = ivec4(0x0, 0x7e00, 0x7e0000, 0x0);
    d[30] = ivec4(0x20, 0x10080402, 0x4081020, 0x0);
    d[31] = ivec4(0x3c42, 0x2040810, 0x1010, 0x0);
    d[32] = ivec4(0x1c22, 0x414f5151, 0x51534d40, 0x201f0000);
    d[33] = ivec4(0x18, 0x24424242, 0x7e424242, 0x0);
    d[34] = ivec4(0x7c, 0x4242427c, 0x4242427c, 0x0);
    d[35] = ivec4(0x1e, 0x20404040, 0x4040201e, 0x0);
    d[36] = ivec4(0x78, 0x44424242, 0x42424478, 0x0);
    d[37] = ivec4(0x7e, 0x4040407c, 0x4040407e, 0x0);
    d[38] = ivec4(0x7e, 0x4040407c, 0x40404040, 0x0);
    d[39] = ivec4(0x1e, 0x20404046, 0x4242221e, 0x0);
    d[40] = ivec4(0x42, 0x4242427e, 0x42424242, 0x0);
    d[41] = ivec4(0x3e, 0x8080808, 0x808083e, 0x0);
    d[42] = ivec4(0x2, 0x2020202, 0x242423c, 0x0);
    d[43] = ivec4(0x42, 0x44485060, 0x50484442, 0x0);
    d[44] = ivec4(0x40, 0x40404040, 0x4040407e, 0x0);
    d[45] = ivec4(0x41, 0x63554949, 0x41414141, 0x0);
    d[46] = ivec4(0x42, 0x62524a46, 0x42424242, 0x0);
    d[47] = ivec4(0x3c, 0x42424242, 0x4242423c, 0x0);
    d[48] = ivec4(0x7c, 0x4242427c, 0x40404040, 0x0);
    d[49] = ivec4(0x3c, 0x42424242, 0x4242423c, 0x4020000);
    d[50] = ivec4(0x7c, 0x4242427c, 0x48444242, 0x0);
    d[51] = ivec4(0x3e, 0x40402018, 0x402027c, 0x0);
    d[52] = ivec4(0x7f, 0x8080808, 0x8080808, 0x0);
    d[53] = ivec4(0x42, 0x42424242, 0x4242423c, 0x0);
    d[54] = ivec4(0x42, 0x42424242, 0x24241818, 0x0);
    d[55] = ivec4(0x41, 0x41414149, 0x49495563, 0x0);
    d[56] = ivec4(0x41, 0x41221408, 0x14224141, 0x0);
    d[57] = ivec4(0x41, 0x41221408, 0x8080808, 0x0);
    d[58] = ivec4(0x7e, 0x4080810, 0x1020207e, 0x0);
    d[59] = ivec4(0x1e10, 0x10101010, 0x10101010, 0x101e0000);
    d[60] = ivec4(0x4040, 0x20201010, 0x8080404, 0x2020000);
    d[61] = ivec4(0x7808, 0x8080808, 0x8080808, 0x8780000);
    d[62] = ivec4(0x1028, 0x44000000, 0x0, 0x0);
    d[63] = ivec4(0x0, 0x0, 0x0, 0xff0000);
    d[64] = ivec4(0x201008, 0x4000000, 0x0, 0x0);
    d[65] = ivec4(0x0, 0x3c0202, 0x3e42423e, 0x0);
    d[66] = ivec4(0x4040, 0x407c4242, 0x4242427c, 0x0);
    d[67] = ivec4(0x0, 0x3c4240, 0x4040423c, 0x0);
    d[68] = ivec4(0x202, 0x23e4242, 0x4242423e, 0x0);
    d[69] = ivec4(0x0, 0x3c4242, 0x7e40403e, 0x0);
    d[70] = ivec4(0xe10, 0x107e1010, 0x10101010, 0x0);
    d[71] = ivec4(0x0, 0x3e4242, 0x4242423e, 0x2023c00);
    d[72] = ivec4(0x4040, 0x407c4242, 0x42424242, 0x0);
    d[73] = ivec4(0x808, 0x380808, 0x808083e, 0x0);
    d[74] = ivec4(0x404, 0x1c0404, 0x4040404, 0x4043800);
    d[75] = ivec4(0x4040, 0x40444850, 0x70484442, 0x0);
    d[76] = ivec4(0x3808, 0x8080808, 0x808083e, 0x0);
    d[77] = ivec4(0x0, 0x774949, 0x49494949, 0x0);
    d[78] = ivec4(0x0, 0x7c4242, 0x42424242, 0x0);
    d[79] = ivec4(0x0, 0x3c4242, 0x4242423c, 0x0);
    d[80] = ivec4(0x0, 0x7c4242, 0x4242427c, 0x40404000);
    d[81] = ivec4(0x0, 0x3e4242, 0x4242423e, 0x2020200);
    d[82] = ivec4(0x0, 0x2e3020, 0x20202020, 0x0);
    d[83] = ivec4(0x0, 0x3e4020, 0x1804027c, 0x0);
    d[84] = ivec4(0x10, 0x107e1010, 0x1010100e, 0x0);
    d[85] = ivec4(0x0, 0x424242, 0x4242423e, 0x0);
    d[86] = ivec4(0x0, 0x424242, 0x24241818, 0x0);
    d[87] = ivec4(0x0, 0x414141, 0x49495563, 0x0);
    d[88] = ivec4(0x0, 0x412214, 0x8142241, 0x0);
    d[89] = ivec4(0x0, 0x424242, 0x4242423e, 0x2023c00);
    d[90] = ivec4(0x0, 0x7e0408, 0x1020407e, 0x0);
    d[91] = ivec4(0xe1010, 0x101010e0, 0x10101010, 0x100e0000);
    d[92] = ivec4(0x80808, 0x8080808, 0x8080808, 0x8080000);
    d[93] = ivec4(0x700808, 0x8080807, 0x8080808, 0x8700000);
    d[94] = ivec4(0x3149, 0x46000000, 0x0, 0x0);
    d[95] = ivec4(0x0, 0x0, 0x0, 0x0);

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < CHAR_TOTAL; i++)
        if (i == index) return d[i];
    return ivec4(0x0, 0x0, 0x0, 0x0);
    #else
    return d[ clamp(index, 0, CHAR_TOTAL) ];
    #endif
}

float char(vec2 uv, int char_code) {
    ivec2 char_coord = ivec2(7, 15) - ivec2(floor(uv * vec2(8.0, 16.0)));
    
    // Pick the correct character bitmap, and then
    // the uint holding covering the four lines that 
    // our y pixel coordinate is in.

    #if defined(PLATFORM_WEBGL)
    ivec4 col = charLUT(char_code);
    int four_lines = col.w;
    int index = char_coord.y/4;
    if (index == 0) four_lines = col.x;
    else if (index == 1) four_lines = col.y;
    else if (index == 2) four_lines = col.z;
    #else
    int four_lines = charLUT(char_code)[char_coord.y/4];
    #endif

    // Now we must pick the correct line
    #if __VERSION__ < 130 || defined(PLATFORM_WEBGL)
    int current_line = modi(four_lines / int(pow(256.0, float(3-modi(char_coord.y,4)))),256);
    int current_pixel = modi(current_line / int(pow(2.0, float(char_coord.x))),2);
    #else
    int current_line  = (four_lines >> (8*(3-(char_coord.y)%4))) & 0xff;
    int current_pixel = (current_line >> (char_coord.x)) & 0x01;
    #endif
    return float(current_pixel);
}

#endif