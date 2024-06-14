/*
contributors: Patricio Gonzalez Vivo
description: | 
    Sanzo Wada's color triads from ["A Dictionary of Color Combinations"](https://sanzo-wada.dmbk.io/)
use: 
    - <int3> wadaTriads (<int> index)
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
    
int3 wadaTriad( const int index ) {
    int3 t[WADA_TRIAD_TOTAL];
    t[0] = int3(2,3,150);
    t[1] = int3(54,61,124);
    t[2] = int3(45,135,136);
    t[3] = int3(38,135,136);
    t[4] = int3(31,49,105);
    t[5] = int3(22,46,98);
    t[6] = int3(5,55,145);
    t[7] = int3(35,59,92);
    t[8] = int3(41,59,124);
    t[9] = int3(45,96,139);
    t[10] = int3(56,69,111);
    t[11] = int3(13,28,147);
    t[12] = int3(13,65,116);
    t[13] = int3(38,47,70);
    t[14] = int3(33,70,97);
    t[15] = int3(38,104,111);
    t[16] = int3(21,101,148);
    t[17] = int3(47,55,114);
    t[18] = int3(111,129,152);
    t[19] = int3(47,120,155);
    t[20] = int3(63,89,127);
    t[21] = int3(26,68,111);
    t[22] = int3(119,134,154);
    t[23] = int3(24,63,156);
    t[24] = int3(69,80,103);
    t[25] = int3(9,33,86);
    t[26] = int3(58,60,115);
    t[27] = int3(58,63,130);
    t[28] = int3(18,32,110);
    t[29] = int3(22,54,119);
    t[30] = int3(15,98,129);
    t[31] = int3(58,112,147);
    t[32] = int3(34,58,118);
    t[33] = int3(55,66,91);
    t[34] = int3(69,113,131);
    t[35] = int3(68,81,122);
    t[36] = int3(56,86,120);
    t[37] = int3(17,60,147);
    t[38] = int3(7,40,130);
    t[39] = int3(1,39,154);
    t[40] = int3(24,60,148);
    t[41] = int3(35,61,87);
    t[42] = int3(66,120,148);
    t[43] = int3(55,83,86);
    t[44] = int3(1,132,145);
    t[45] = int3(44,52,140);
    t[46] = int3(0,42,113);
    t[47] = int3(35,71,131);
    t[48] = int3(41,110,117);
    t[49] = int3(17,47,124);
    t[50] = int3(28,147,149);
    t[51] = int3(13,85,129);
    t[52] = int3(142,143,148);
    t[53] = int3(9,41,73);
    t[54] = int3(4,18,39);
    t[55] = int3(32,49,119);
    t[56] = int3(121,132,137);
    t[57] = int3(55,107,114);
    t[58] = int3(41,65,156);
    t[59] = int3(4,59,119);
    t[60] = int3(46,85,142);
    t[61] = int3(7,40,99);
    t[62] = int3(15,42,118);
    t[63] = int3(9,39,152);
    t[64] = int3(8,139,152);
    t[65] = int3(20,79,124);
    t[66] = int3(28,52,95);
    t[67] = int3(7,75,112);
    t[68] = int3(86,112,155);
    t[69] = int3(39,48,105);
    t[70] = int3(24,66,110);
    t[71] = int3(35,48,147);
    t[72] = int3(77,87,156);
    t[73] = int3(38,117,120);
    t[74] = int3(41,61,111);
    t[75] = int3(45,55,105);
    t[76] = int3(62,92,129);
    t[77] = int3(29,76,111);
    t[78] = int3(48,56,109);
    t[79] = int3(41,77,147);
    t[80] = int3(15,100,156);
    t[81] = int3(35,49,103);
    t[82] = int3(121,131,142);
    t[83] = int3(15,95,123);
    t[84] = int3(25,49,137);
    t[85] = int3(28,152,156);
    t[86] = int3(59,61,67);
    t[87] = int3(4,49,86);
    t[88] = int3(9,122,145);
    t[89] = int3(22,123,144);
    t[90] = int3(37,46,147);
    t[91] = int3(0,110,115);
    t[92] = int3(28,39,152);
    t[93] = int3(7,86,112);
    t[94] = int3(22,44,129);
    t[95] = int3(28,71,125);
    t[96] = int3(45,81,109);
    t[97] = int3(41,61,132);
    t[98] = int3(69,124,141);
    t[99] = int3(49,149,154);
    t[100] = int3(4,57,87);
    t[101] = int3(6,54,115);
    t[102] = int3(16,41,98);
    t[103] = int3(13,55,64);
    t[104] = int3(16,41,137);
    t[105] = int3(44,56,86);
    t[106] = int3(16,27,140);
    t[107] = int3(45,83,156);
    t[108] = int3(27,62,99);
    t[109] = int3(60,123,132);
    t[110] = int3(19,41,80);
    t[111] = int3(86,90,94);

    return t[index];
}
#endif