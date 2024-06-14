/*
contributors: Patricio Gonzalez Vivo
description: | 
    Sanzo Wada's color tetrads from ["A Dictionary of Color Combinations"](https://sanzo-wada.dmbk.io/)
use:
    - <ivec4> wadaTetrads (<int> index)
defines:
    - WADA_TETRAD_TOTAL
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wada_tetrads.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef WADA_TETRAD_TOTAL
#define WADA_TETRAD_TOTAL 99
#endif

#ifndef FNC_PALETTE_WADA_TETRAD
#define FNC_PALETTE_WADA_TETRAD
                
ivec4 wadaTetrad( const int index ) {
    ivec4 t[WADA_TETRAD_TOTAL];
    t[0] = ivec4(2,3,115,125);
    t[1] = ivec4(1,114,135,136);
    t[2] = ivec4(8,135,136,148);
    t[3] = ivec4(45,135,136,152);
    t[4] = ivec4(19,56,100,156);
    t[5] = ivec4(28,83,96,111);
    t[6] = ivec4(17,39,50,122);
    t[7] = ivec4(8,19,103,156);
    t[8] = ivec4(13,41,78,155);
    t[9] = ivec4(4,14,94,120);
    t[10] = ivec4(28,102,127,155);
    t[11] = ivec4(1,27,38,45);
    t[12] = ivec4(13,56,98,124);
    t[13] = ivec4(8,69,131,144);
    t[14] = ivec4(32,53,58,122);
    t[15] = ivec4(57,64,97,108);
    t[16] = ivec4(30,54,103,155);
    t[17] = ivec4(12,13,68,117);
    t[18] = ivec4(55,62,149,155);
    t[19] = ivec4(1,38,75,133);
    t[20] = ivec4(13,57,113,156);
    t[21] = ivec4(48,63,101,156);
    t[22] = ivec4(16,60,119,137);
    t[23] = ivec4(35,44,75,120);
    t[24] = ivec4(55,117,121,154);
    t[25] = ivec4(10,48,87,97);
    t[26] = ivec4(30,39,113,154);
    t[27] = ivec4(12,41,70,104);
    t[28] = ivec4(19,44,86,155);
    t[29] = ivec4(1,17,88,115);
    t[30] = ivec4(10,56,92,128);
    t[31] = ivec4(46,61,98,119);
    t[32] = ivec4(4,13,107,108);
    t[33] = ivec4(13,35,137,156);
    t[34] = ivec4(12,38,78,104);
    t[35] = ivec4(25,112,117,130);
    t[36] = ivec4(39,63,86,111);
    t[37] = ivec4(0,34,77,152);
    t[38] = ivec4(18,53,83,145);
    t[39] = ivec4(8,42,89,156);
    t[40] = ivec4(9,24,75,128);
    t[41] = ivec4(46,58,104,123);
    t[42] = ivec4(13,48,53,127);
    t[43] = ivec4(12,105,133,145);
    t[44] = ivec4(39,98,112,120);
    t[45] = ivec4(12,51,112,134);
    t[46] = ivec4(20,35,95,114);
    t[47] = ivec4(11,56,97,123);
    t[48] = ivec4(4,19,64,86);
    t[49] = ivec4(19,60,116,125);
    t[50] = ivec4(8,57,109,113);
    t[51] = ivec4(61,82,145,156);
    t[52] = ivec4(55,90,125,128);
    t[53] = ivec4(37,39,105,112);
    t[54] = ivec4(90,97,112,113);
    t[55] = ivec4(39,44,50,53);
    t[56] = ivec4(13,86,93,100);
    t[57] = ivec4(38,46,104,111);
    t[58] = ivec4(46,54,119,128);
    t[59] = ivec4(38,49,81,155);
    t[60] = ivec4(19,61,78,125);
    t[61] = ivec4(23,48,102,120);
    t[62] = ivec4(7,46,86,113);
    t[63] = ivec4(46,53,108,120);
    t[64] = ivec4(40,64,107,152);
    t[65] = ivec4(32,46,72,98);
    t[66] = ivec4(55,98,101,108);
    t[67] = ivec4(22,131,137,149);
    t[68] = ivec4(38,44,75,107);
    t[69] = ivec4(29,46,88,90);
    t[70] = ivec4(19,61,93,121);
    t[71] = ivec4(22,54,103,156);
    t[72] = ivec4(8,32,124,150);
    t[73] = ivec4(33,101,147,148);
    t[74] = ivec4(43,53,55,86);
    t[75] = ivec4(79,106,123,130);
    t[76] = ivec4(13,56,61,104);
    t[77] = ivec4(5,38,102,110);
    t[78] = ivec4(4,38,107,111);
    t[79] = ivec4(29,118,137,152);
    t[80] = ivec4(12,40,59,130);
    t[81] = ivec4(38,64,89,91);
    t[82] = ivec4(8,13,53,132);
    t[83] = ivec4(46,132,149,155);
    t[84] = ivec4(52,108,111,117);
    t[85] = ivec4(5,21,107,123);
    t[86] = ivec4(19,55,112,119);
    t[87] = ivec4(42,75,89,120);
    t[88] = ivec4(33,61,147,155);
    t[89] = ivec4(8,32,39,106);
    t[90] = ivec4(37,133,146,156);
    t[91] = ivec4(49,65,110,125);
    t[92] = ivec4(64,97,152,156);
    t[93] = ivec4(7,104,107,110);
    t[94] = ivec4(1,46,76,107);
    t[95] = ivec4(45,124,137,156);
    t[96] = ivec4(32,108,114,140);
    t[97] = ivec4(92,97,121,134);
    t[98] = ivec4(52,104,107,149);

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < WADA_TETRAD_TOTAL; i++)
        if (i == index) return t[i];
    #else
    return t[index];
    #endif                   
}
#endif
