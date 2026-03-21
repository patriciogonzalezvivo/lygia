#include "../math/modi.wgsl"

/*
contributors: ['Tim Gfrerer', 'Patricio Gonzalez Vivo']
description: |
    Draws a character from a bitmap font. 
    Based on Tim's article about Texture-less Text Rendering https://poniesandlight.co.uk/reflect/debug_print_text/
use: <vec2> char(<vec2> uv, <int> char_code)
*/

// #define CHAR_SIZE vec2(.02)

const CHAR_TOTAL: f32 = 96;

const CHAR_SPACE: f32 = 0;
const CHAR_EXCLAMATION: f32 = 1;
const CHAR_QUOTE: f32 = 2;
const CHAR_NUMBER: f32 = 3;
const CHAR_DOLLAR: f32 = 4;
const CHAR_PERCENT: f32 = 5;
const CHAR_AMPERSAND: f32 = 6;
const CHAR_APOSTROPHE: f32 = 7;
const CHAR_PAREN_LEFT: f32 = 8;
const CHAR_PAREN_RIGHT: f32 = 9;
const CHAR_ASTERISK: f32 = 10;
const CHAR_PLUS: f32 = 11;
const CHAR_COMMA: f32 = 12;
const CHAR_MINUS: f32 = 13;
const CHAR_PERIOD: f32 = 14;
const CHAR_SLASH: f32 = 15;
const CHAR_0: f32 = 16;
const CHAR_1: f32 = 17;
const CHAR_2: f32 = 18;
const CHAR_3: f32 = 19;
const CHAR_4: f32 = 20;
const CHAR_5: f32 = 21;
const CHAR_6: f32 = 22;
const CHAR_7: f32 = 23;
const CHAR_8: f32 = 24;
const CHAR_9: f32 = 25;
const CHAR_COLON: f32 = 26;
const CHAR_SEMICOLON: f32 = 27;
const CHAR_LESS: f32 = 28;
const CHAR_EQUAL: f32 = 29;
const CHAR_GREATER: f32 = 30;
const CHAR_QUESTION: f32 = 31;
const CHAR_AT: f32 = 32;
const CHAR_A: f32 = 33;
const CHAR_B: f32 = 34;
const CHAR_C: f32 = 35;
const CHAR_D: f32 = 36;
const CHAR_E: f32 = 37;
const CHAR_F: f32 = 38;
const CHAR_G: f32 = 39;
const CHAR_H: f32 = 40;
const CHAR_I: f32 = 41;
const CHAR_J: f32 = 42;
const CHAR_K: f32 = 43;
const CHAR_L: f32 = 44;
const CHAR_M: f32 = 45;
const CHAR_N: f32 = 46;
const CHAR_O: f32 = 47;
const CHAR_P: f32 = 48;
const CHAR_Q: f32 = 49;
const CHAR_R: f32 = 50;
const CHAR_S: f32 = 51;
const CHAR_T: f32 = 52;
const CHAR_U: f32 = 53;
const CHAR_V: f32 = 54;
const CHAR_W: f32 = 55;
const CHAR_X: f32 = 56;
const CHAR_Y: f32 = 57;
const CHAR_Z: f32 = 58;
const CHAR_BRACKET_LEFT: f32 = 59;
const CHAR_BACKSLASH: f32 = 60;
const CHAR_BRACKET_RIGHT: f32 = 61;
const CHAR_CARET: f32 = 62;
const CHAR_UNDERSCORE: f32 = 63;
const CHAR_GRAVE: f32 = 64;
const CHAR_a: f32 = 65;
const CHAR_b: f32 = 66;
const CHAR_c: f32 = 67;
const CHAR_d: f32 = 68;
const CHAR_e: f32 = 69;
const CHAR_f: f32 = 70;
const CHAR_g: f32 = 71;
const CHAR_h: f32 = 72;
const CHAR_i: f32 = 73;
const CHAR_j: f32 = 74;
const CHAR_k: f32 = 75;
const CHAR_l: f32 = 76;
const CHAR_m: f32 = 77;
const CHAR_n: f32 = 78;
const CHAR_o: f32 = 79;
const CHAR_p: f32 = 80;
const CHAR_q: f32 = 81;
const CHAR_r: f32 = 82;
const CHAR_s: f32 = 83;
const CHAR_t: f32 = 84;
const CHAR_u: f32 = 85;
const CHAR_v: f32 = 86;
const CHAR_w: f32 = 87;
const CHAR_x: f32 = 88;
const CHAR_y: f32 = 89;
const CHAR_z: f32 = 90;
const CHAR_BRACE_LEFT: f32 = 91;
const CHAR_BAR: f32 = 92;
const CHAR_BRACE_RIGHT: f32 = 93;
const CHAR_TILDE: f32 = 94;

fn charLUT(index: i32) -> vec4i {
    ivec4 d[CHAR_TOTAL];
    d[0] = vec4i(0x0, 0x0, 0x0, 0x0);
    d[1] = vec4i(0x1010, 0x10101010, 0x1010, 0x0);
    d[2] = vec4i(0x242424, 0x24000000, 0x0, 0x0);
    d[3] = vec4i(0x24, 0x247e2424, 0x247e2424, 0x0);
    d[4] = vec4i(0x808, 0x1e20201c, 0x2023c08, 0x8000000);
    d[5] = vec4i(0x30, 0x494a3408, 0x16294906, 0x0);
    d[6] = vec4i(0x3048, 0x48483031, 0x49464639, 0x0);
    d[7] = vec4i(0x101010, 0x10000000, 0x0, 0x0);
    d[8] = vec4i(0x408, 0x8101010, 0x10101008, 0x8040000);
    d[9] = vec4i(0x2010, 0x10080808, 0x8080810, 0x10200000);
    d[10] = vec4i(0x0, 0x24187e, 0x18240000, 0x0);
    d[11] = vec4i(0x0, 0x808087f, 0x8080800, 0x0);
    d[12] = vec4i(0x0, 0x0, 0x1818, 0x8081000);
    d[13] = vec4i(0x0, 0x7e, 0x0, 0x0);
    d[14] = vec4i(0x0, 0x0, 0x1818, 0x0);
    d[15] = vec4i(0x202, 0x4040808, 0x10102020, 0x40400000);
    d[16] = vec4i(0x3c, 0x42464a52, 0x6242423c, 0x0);
    d[17] = vec4i(0x8, 0x18280808, 0x808083e, 0x0);
    d[18] = vec4i(0x3c, 0x42020204, 0x810207e, 0x0);
    d[19] = vec4i(0x7e, 0x4081c02, 0x202423c, 0x0);
    d[20] = vec4i(0x4, 0xc142444, 0x7e040404, 0x0);
    d[21] = vec4i(0x7e, 0x40407c02, 0x202423c, 0x0);
    d[22] = vec4i(0x1c, 0x2040407c, 0x4242423c, 0x0);
    d[23] = vec4i(0x7e, 0x2040408, 0x8101010, 0x0);
    d[24] = vec4i(0x3c, 0x4242423c, 0x4242423c, 0x0);
    d[25] = vec4i(0x3c, 0x4242423e, 0x2020438, 0x0);
    d[26] = vec4i(0x0, 0x181800, 0x1818, 0x0);
    d[27] = vec4i(0x0, 0x181800, 0x1818, 0x8081000);
    d[28] = vec4i(0x4, 0x8102040, 0x20100804, 0x0);
    d[29] = vec4i(0x0, 0x7e00, 0x7e0000, 0x0);
    d[30] = vec4i(0x20, 0x10080402, 0x4081020, 0x0);
    d[31] = vec4i(0x3c42, 0x2040810, 0x1010, 0x0);
    d[32] = vec4i(0x1c22, 0x414f5151, 0x51534d40, 0x201f0000);
    d[33] = vec4i(0x18, 0x24424242, 0x7e424242, 0x0);
    d[34] = vec4i(0x7c, 0x4242427c, 0x4242427c, 0x0);
    d[35] = vec4i(0x1e, 0x20404040, 0x4040201e, 0x0);
    d[36] = vec4i(0x78, 0x44424242, 0x42424478, 0x0);
    d[37] = vec4i(0x7e, 0x4040407c, 0x4040407e, 0x0);
    d[38] = vec4i(0x7e, 0x4040407c, 0x40404040, 0x0);
    d[39] = vec4i(0x1e, 0x20404046, 0x4242221e, 0x0);
    d[40] = vec4i(0x42, 0x4242427e, 0x42424242, 0x0);
    d[41] = vec4i(0x3e, 0x8080808, 0x808083e, 0x0);
    d[42] = vec4i(0x2, 0x2020202, 0x242423c, 0x0);
    d[43] = vec4i(0x42, 0x44485060, 0x50484442, 0x0);
    d[44] = vec4i(0x40, 0x40404040, 0x4040407e, 0x0);
    d[45] = vec4i(0x41, 0x63554949, 0x41414141, 0x0);
    d[46] = vec4i(0x42, 0x62524a46, 0x42424242, 0x0);
    d[47] = vec4i(0x3c, 0x42424242, 0x4242423c, 0x0);
    d[48] = vec4i(0x7c, 0x4242427c, 0x40404040, 0x0);
    d[49] = vec4i(0x3c, 0x42424242, 0x4242423c, 0x4020000);
    d[50] = vec4i(0x7c, 0x4242427c, 0x48444242, 0x0);
    d[51] = vec4i(0x3e, 0x40402018, 0x402027c, 0x0);
    d[52] = vec4i(0x7f, 0x8080808, 0x8080808, 0x0);
    d[53] = vec4i(0x42, 0x42424242, 0x4242423c, 0x0);
    d[54] = vec4i(0x42, 0x42424242, 0x24241818, 0x0);
    d[55] = vec4i(0x41, 0x41414149, 0x49495563, 0x0);
    d[56] = vec4i(0x41, 0x41221408, 0x14224141, 0x0);
    d[57] = vec4i(0x41, 0x41221408, 0x8080808, 0x0);
    d[58] = vec4i(0x7e, 0x4080810, 0x1020207e, 0x0);
    d[59] = vec4i(0x1e10, 0x10101010, 0x10101010, 0x101e0000);
    d[60] = vec4i(0x4040, 0x20201010, 0x8080404, 0x2020000);
    d[61] = vec4i(0x7808, 0x8080808, 0x8080808, 0x8780000);
    d[62] = vec4i(0x1028, 0x44000000, 0x0, 0x0);
    d[63] = vec4i(0x0, 0x0, 0x0, 0xff0000);
    d[64] = vec4i(0x201008, 0x4000000, 0x0, 0x0);
    d[65] = vec4i(0x0, 0x3c0202, 0x3e42423e, 0x0);
    d[66] = vec4i(0x4040, 0x407c4242, 0x4242427c, 0x0);
    d[67] = vec4i(0x0, 0x3c4240, 0x4040423c, 0x0);
    d[68] = vec4i(0x202, 0x23e4242, 0x4242423e, 0x0);
    d[69] = vec4i(0x0, 0x3c4242, 0x7e40403e, 0x0);
    d[70] = vec4i(0xe10, 0x107e1010, 0x10101010, 0x0);
    d[71] = vec4i(0x0, 0x3e4242, 0x4242423e, 0x2023c00);
    d[72] = vec4i(0x4040, 0x407c4242, 0x42424242, 0x0);
    d[73] = vec4i(0x808, 0x380808, 0x808083e, 0x0);
    d[74] = vec4i(0x404, 0x1c0404, 0x4040404, 0x4043800);
    d[75] = vec4i(0x4040, 0x40444850, 0x70484442, 0x0);
    d[76] = vec4i(0x3808, 0x8080808, 0x808083e, 0x0);
    d[77] = vec4i(0x0, 0x774949, 0x49494949, 0x0);
    d[78] = vec4i(0x0, 0x7c4242, 0x42424242, 0x0);
    d[79] = vec4i(0x0, 0x3c4242, 0x4242423c, 0x0);
    d[80] = vec4i(0x0, 0x7c4242, 0x4242427c, 0x40404000);
    d[81] = vec4i(0x0, 0x3e4242, 0x4242423e, 0x2020200);
    d[82] = vec4i(0x0, 0x2e3020, 0x20202020, 0x0);
    d[83] = vec4i(0x0, 0x3e4020, 0x1804027c, 0x0);
    d[84] = vec4i(0x10, 0x107e1010, 0x1010100e, 0x0);
    d[85] = vec4i(0x0, 0x424242, 0x4242423e, 0x0);
    d[86] = vec4i(0x0, 0x424242, 0x24241818, 0x0);
    d[87] = vec4i(0x0, 0x414141, 0x49495563, 0x0);
    d[88] = vec4i(0x0, 0x412214, 0x8142241, 0x0);
    d[89] = vec4i(0x0, 0x424242, 0x4242423e, 0x2023c00);
    d[90] = vec4i(0x0, 0x7e0408, 0x1020407e, 0x0);
    d[91] = vec4i(0xe1010, 0x101010e0, 0x10101010, 0x100e0000);
    d[92] = vec4i(0x80808, 0x8080808, 0x8080808, 0x8080000);
    d[93] = vec4i(0x700808, 0x8080807, 0x8080808, 0x8700000);
    d[94] = vec4i(0x3149, 0x46000000, 0x0, 0x0);
    d[95] = vec4i(0x0, 0x0, 0x0, 0x0);

    for (int i = 0; i < CHAR_TOTAL; i++)
        if (i == index) return d[i];
    return vec4i(0x0, 0x0, 0x0, 0x0);
    return d[ clamp(index, 0, CHAR_TOTAL) ];
}

fn char(uv: vec2f, char_code: i32) -> f32 {
    let char_coord = vec2i(7, 15) - vec2i(floor(uv * vec2f(8.0, 16.0)));
    
    // Pick the correct character bitmap, and then
    // the uint holding covering the four lines that 
    // our y pixel coordinate is in.

    let col = charLUT(char_code);
    let four_lines = col.w;
    let index = char_coord.y/4;
    if (index == 0) four_lines = col.x;
    else if (index == 1) four_lines = col.y;
    else if (index == 2) four_lines = col.z;
    let four_lines = charLUT(char_code)[char_coord.y/4];

    // Now we must pick the correct line
    let current_line = modi(four_lines / int(pow(256.0, float(3-modi(char_coord.y,4)))),256);
    let current_pixel = modi(current_line / int(pow(2.0, float(char_coord.x))),2);
    let current_line = (four_lines >> (8*(3-(char_coord.y)%4))) & 0xff;
    let current_pixel = (current_line >> (char_coord.x)) & 0x01;
    return float(current_pixel);
}
