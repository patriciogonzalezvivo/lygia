/*
contributors: Patricio Gonzalez Vivo
description: |
    Sanzo Wada's color dyad. from ["A Dictionary of Color Combinations"](https://sanzo-wada.dmbk.io/)
use:
    - <int2> wadaDuads (<int> index)
defines:
    - WADA_DYAD_TOTAL
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wada_dyads.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef WADA_DYAD_TOTAL
#define WADA_DYAD_TOTAL 124
#endif

#ifndef FNC_PALETTE_WADA_DUAD
#define FNC_PALETTE_WADA_DUAD            
int2 wadaDyad( const int index ) {
    int2 d[WADA_DYAD_TOTAL];
    d[0] = int2(65,115);
    d[1] = int2(13,39);
    d[2] = int2(50,148);
    d[3] = int2(7,129);
    d[4] = int2(63,87);
    d[5] = int2(66,131);
    d[6] = int2(36,138);
    d[7] = int2(66,72);
    d[8] = int2(41,153);
    d[9] = int2(50,117);
    d[10] = int2(13,143);
    d[11] = int2(9,40);
    d[12] = int2(98,131);
    d[13] = int2(33,109);
    d[14] = int2(12,97);
    d[15] = int2(84,91);
    d[16] = int2(113,133);
    d[17] = int2(7,97);
    d[18] = int2(82,143);
    d[19] = int2(18,108);
    d[20] = int2(47,81);
    d[21] = int2(1,155);
    d[22] = int2(83,129);
    d[23] = int2(74,111);
    d[24] = int2(29,153);
    d[25] = int2(17,39);
    d[26] = int2(49,91);
    d[27] = int2(13,155);
    d[28] = int2(8,152);
    d[29] = int2(4,28);
    d[30] = int2(68,86);
    d[31] = int2(27,148);
    d[32] = int2(103,124);
    d[33] = int2(22,121);
    d[34] = int2(72,113);
    d[35] = int2(59,147);
    d[36] = int2(1,137);
    d[37] = int2(99,118);
    d[38] = int2(42,55);
    d[39] = int2(63,156);
    d[40] = int2(18,132);
    d[41] = int2(24,121);
    d[42] = int2(109,119);
    d[43] = int2(41,150);
    d[44] = int2(28,119);
    d[45] = int2(38,156);
    d[46] = int2(61,150);
    d[47] = int2(98,110);
    d[48] = int2(10,151);
    d[49] = int2(132,147);
    d[50] = int2(145,155);
    d[51] = int2(32,97);
    d[52] = int2(8,70);
    d[53] = int2(39,127);
    d[54] = int2(90,149);
    d[55] = int2(54,156);
    d[56] = int2(37,115);
    d[57] = int2(137,139);
    d[58] = int2(68,113);
    d[59] = int2(58,78);
    d[60] = int2(118,127);
    d[61] = int2(4,54);
    d[62] = int2(154,156);
    d[63] = int2(13,105);
    d[64] = int2(29,49);
    d[65] = int2(38,109);
    d[66] = int2(86,117);
    d[67] = int2(109,125);
    d[68] = int2(39,154);
    d[69] = int2(44,114);
    d[70] = int2(83,117);
    d[71] = int2(47,154);
    d[72] = int2(32,150);
    d[73] = int2(52,125);
    d[74] = int2(42,130);
    d[75] = int2(14,120);
    d[76] = int2(13,97);
    d[77] = int2(42,119);
    d[78] = int2(61,125);
    d[79] = int2(8,137);
    d[80] = int2(37,67);
    d[81] = int2(5,98);
    d[82] = int2(70,110);
    d[83] = int2(41,123);
    d[84] = int2(32,128);
    d[85] = int2(59,75);
    d[86] = int2(1,18);
    d[87] = int2(83,125);
    d[88] = int2(39,115);
    d[89] = int2(71,138);
    d[90] = int2(41,67);
    d[91] = int2(66,150);
    d[92] = int2(28,38);
    d[93] = int2(120,128);
    d[94] = int2(56,73);
    d[95] = int2(8,27);
    d[96] = int2(39,106);
    d[97] = int2(31,85);
    d[98] = int2(39,89);
    d[99] = int2(7,156);
    d[100] = int2(42,85);
    d[101] = int2(60,120);
    d[102] = int2(40,64);
    d[103] = int2(22,156);
    d[104] = int2(59,85);
    d[105] = int2(110,127);
    d[106] = int2(18,45);
    d[107] = int2(31,128);
    d[108] = int2(42,87);
    d[109] = int2(38,61);
    d[110] = int2(8,60);
    d[111] = int2(31,121);
    d[112] = int2(10,134);
    d[113] = int2(9,37);
    d[114] = int2(53,109);
    d[115] = int2(55,143);
    d[116] = int2(111,112);
    d[117] = int2(109,140);
    d[118] = int2(1,66);
    d[119] = int2(46,119);
    d[120] = int2(107,152);
    d[121] = int2(32,118);
    d[122] = int2(62,125);
    d[123] = int2(7,38);

    return d[index];
}
#endif