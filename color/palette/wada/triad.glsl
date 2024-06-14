/*
contributors: Patricio Gonzalez Vivo
description: | 
    Sanzo Wada's color triads from ["A Dictionary of Color Combinations"](https://sanzo-wada.dmbk.io/)
use: 
    - <ivec3> wadaTriads (<int> index)
defines:
    - WADA_TRIAD_TOTAL
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wada_triads.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef WADA_TRIAD_TOTAL
#define WADA_TRIAD_TOTAL 112
#endif

#ifndef FNC_PALETTE_WADA_TRIAD
#define FNC_PALETTE_WADA_TRIAD
    
ivec3 wadaTriad( const int index ) {
    ivec3 t[WADA_TRIAD_TOTAL];
    t[0] = ivec3(2,3,150);
    t[1] = ivec3(54,61,124);
    t[2] = ivec3(45,135,136);
    t[3] = ivec3(38,135,136);
    t[4] = ivec3(31,49,105);
    t[5] = ivec3(22,46,98);
    t[6] = ivec3(5,55,145);
    t[7] = ivec3(35,59,92);
    t[8] = ivec3(41,59,124);
    t[9] = ivec3(45,96,139);
    t[10] = ivec3(56,69,111);
    t[11] = ivec3(13,28,147);
    t[12] = ivec3(13,65,116);
    t[13] = ivec3(38,47,70);
    t[14] = ivec3(33,70,97);
    t[15] = ivec3(38,104,111);
    t[16] = ivec3(21,101,148);
    t[17] = ivec3(47,55,114);
    t[18] = ivec3(111,129,152);
    t[19] = ivec3(47,120,155);
    t[20] = ivec3(63,89,127);
    t[21] = ivec3(26,68,111);
    t[22] = ivec3(119,134,154);
    t[23] = ivec3(24,63,156);
    t[24] = ivec3(69,80,103);
    t[25] = ivec3(9,33,86);
    t[26] = ivec3(58,60,115);
    t[27] = ivec3(58,63,130);
    t[28] = ivec3(18,32,110);
    t[29] = ivec3(22,54,119);
    t[30] = ivec3(15,98,129);
    t[31] = ivec3(58,112,147);
    t[32] = ivec3(34,58,118);
    t[33] = ivec3(55,66,91);
    t[34] = ivec3(69,113,131);
    t[35] = ivec3(68,81,122);
    t[36] = ivec3(56,86,120);
    t[37] = ivec3(17,60,147);
    t[38] = ivec3(7,40,130);
    t[39] = ivec3(1,39,154);
    t[40] = ivec3(24,60,148);
    t[41] = ivec3(35,61,87);
    t[42] = ivec3(66,120,148);
    t[43] = ivec3(55,83,86);
    t[44] = ivec3(1,132,145);
    t[45] = ivec3(44,52,140);
    t[46] = ivec3(0,42,113);
    t[47] = ivec3(35,71,131);
    t[48] = ivec3(41,110,117);
    t[49] = ivec3(17,47,124);
    t[50] = ivec3(28,147,149);
    t[51] = ivec3(13,85,129);
    t[52] = ivec3(142,143,148);
    t[53] = ivec3(9,41,73);
    t[54] = ivec3(4,18,39);
    t[55] = ivec3(32,49,119);
    t[56] = ivec3(121,132,137);
    t[57] = ivec3(55,107,114);
    t[58] = ivec3(41,65,156);
    t[59] = ivec3(4,59,119);
    t[60] = ivec3(46,85,142);
    t[61] = ivec3(7,40,99);
    t[62] = ivec3(15,42,118);
    t[63] = ivec3(9,39,152);
    t[64] = ivec3(8,139,152);
    t[65] = ivec3(20,79,124);
    t[66] = ivec3(28,52,95);
    t[67] = ivec3(7,75,112);
    t[68] = ivec3(86,112,155);
    t[69] = ivec3(39,48,105);
    t[70] = ivec3(24,66,110);
    t[71] = ivec3(35,48,147);
    t[72] = ivec3(77,87,156);
    t[73] = ivec3(38,117,120);
    t[74] = ivec3(41,61,111);
    t[75] = ivec3(45,55,105);
    t[76] = ivec3(62,92,129);
    t[77] = ivec3(29,76,111);
    t[78] = ivec3(48,56,109);
    t[79] = ivec3(41,77,147);
    t[80] = ivec3(15,100,156);
    t[81] = ivec3(35,49,103);
    t[82] = ivec3(121,131,142);
    t[83] = ivec3(15,95,123);
    t[84] = ivec3(25,49,137);
    t[85] = ivec3(28,152,156);
    t[86] = ivec3(59,61,67);
    t[87] = ivec3(4,49,86);
    t[88] = ivec3(9,122,145);
    t[89] = ivec3(22,123,144);
    t[90] = ivec3(37,46,147);
    t[91] = ivec3(0,110,115);
    t[92] = ivec3(28,39,152);
    t[93] = ivec3(7,86,112);
    t[94] = ivec3(22,44,129);
    t[95] = ivec3(28,71,125);
    t[96] = ivec3(45,81,109);
    t[97] = ivec3(41,61,132);
    t[98] = ivec3(69,124,141);
    t[99] = ivec3(49,149,154);
    t[100] = ivec3(4,57,87);
    t[101] = ivec3(6,54,115);
    t[102] = ivec3(16,41,98);
    t[103] = ivec3(13,55,64);
    t[104] = ivec3(16,41,137);
    t[105] = ivec3(44,56,86);
    t[106] = ivec3(16,27,140);
    t[107] = ivec3(45,83,156);
    t[108] = ivec3(27,62,99);
    t[109] = ivec3(60,123,132);
    t[110] = ivec3(19,41,80);
    t[111] = ivec3(86,90,94);

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < WADA_TRIAD_TOTAL; i++)
        if (i == index) return t[i];
    #else
    return t[index];
    #endif
}
#endif
