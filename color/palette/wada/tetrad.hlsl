/*
contributors: Patricio Gonzalez Vivo
description: | 
    Sanzo Wada's color tetrads from ["A Dictionary of Color Combinations"](https://sanzo-wada.dmbk.io/)
use:
    - <int4> wadaTetrads (<int> index)
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
                
int4 wadaTetrad( const int index ) {
    int4 t[WADA_TETRAD_TOTAL];
    t[0] = int4(2,3,115,125);
    t[1] = int4(1,114,135,136);
    t[2] = int4(8,135,136,148);
    t[3] = int4(45,135,136,152);
    t[4] = int4(19,56,100,156);
    t[5] = int4(28,83,96,111);
    t[6] = int4(17,39,50,122);
    t[7] = int4(8,19,103,156);
    t[8] = int4(13,41,78,155);
    t[9] = int4(4,14,94,120);
    t[10] = int4(28,102,127,155);
    t[11] = int4(1,27,38,45);
    t[12] = int4(13,56,98,124);
    t[13] = int4(8,69,131,144);
    t[14] = int4(32,53,58,122);
    t[15] = int4(57,64,97,108);
    t[16] = int4(30,54,103,155);
    t[17] = int4(12,13,68,117);
    t[18] = int4(55,62,149,155);
    t[19] = int4(1,38,75,133);
    t[20] = int4(13,57,113,156);
    t[21] = int4(48,63,101,156);
    t[22] = int4(16,60,119,137);
    t[23] = int4(35,44,75,120);
    t[24] = int4(55,117,121,154);
    t[25] = int4(10,48,87,97);
    t[26] = int4(30,39,113,154);
    t[27] = int4(12,41,70,104);
    t[28] = int4(19,44,86,155);
    t[29] = int4(1,17,88,115);
    t[30] = int4(10,56,92,128);
    t[31] = int4(46,61,98,119);
    t[32] = int4(4,13,107,108);
    t[33] = int4(13,35,137,156);
    t[34] = int4(12,38,78,104);
    t[35] = int4(25,112,117,130);
    t[36] = int4(39,63,86,111);
    t[37] = int4(0,34,77,152);
    t[38] = int4(18,53,83,145);
    t[39] = int4(8,42,89,156);
    t[40] = int4(9,24,75,128);
    t[41] = int4(46,58,104,123);
    t[42] = int4(13,48,53,127);
    t[43] = int4(12,105,133,145);
    t[44] = int4(39,98,112,120);
    t[45] = int4(12,51,112,134);
    t[46] = int4(20,35,95,114);
    t[47] = int4(11,56,97,123);
    t[48] = int4(4,19,64,86);
    t[49] = int4(19,60,116,125);
    t[50] = int4(8,57,109,113);
    t[51] = int4(61,82,145,156);
    t[52] = int4(55,90,125,128);
    t[53] = int4(37,39,105,112);
    t[54] = int4(90,97,112,113);
    t[55] = int4(39,44,50,53);
    t[56] = int4(13,86,93,100);
    t[57] = int4(38,46,104,111);
    t[58] = int4(46,54,119,128);
    t[59] = int4(38,49,81,155);
    t[60] = int4(19,61,78,125);
    t[61] = int4(23,48,102,120);
    t[62] = int4(7,46,86,113);
    t[63] = int4(46,53,108,120);
    t[64] = int4(40,64,107,152);
    t[65] = int4(32,46,72,98);
    t[66] = int4(55,98,101,108);
    t[67] = int4(22,131,137,149);
    t[68] = int4(38,44,75,107);
    t[69] = int4(29,46,88,90);
    t[70] = int4(19,61,93,121);
    t[71] = int4(22,54,103,156);
    t[72] = int4(8,32,124,150);
    t[73] = int4(33,101,147,148);
    t[74] = int4(43,53,55,86);
    t[75] = int4(79,106,123,130);
    t[76] = int4(13,56,61,104);
    t[77] = int4(5,38,102,110);
    t[78] = int4(4,38,107,111);
    t[79] = int4(29,118,137,152);
    t[80] = int4(12,40,59,130);
    t[81] = int4(38,64,89,91);
    t[82] = int4(8,13,53,132);
    t[83] = int4(46,132,149,155);
    t[84] = int4(52,108,111,117);
    t[85] = int4(5,21,107,123);
    t[86] = int4(19,55,112,119);
    t[87] = int4(42,75,89,120);
    t[88] = int4(33,61,147,155);
    t[89] = int4(8,32,39,106);
    t[90] = int4(37,133,146,156);
    t[91] = int4(49,65,110,125);
    t[92] = int4(64,97,152,156);
    t[93] = int4(7,104,107,110);
    t[94] = int4(1,46,76,107);
    t[95] = int4(45,124,137,156);
    t[96] = int4(32,108,114,140);
    t[97] = int4(92,97,121,134);
    t[98] = int4(52,104,107,149);

    return t[index];
}
#endif