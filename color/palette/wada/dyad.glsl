/*
contributors: Patricio Gonzalez Vivo
description: |
    Sanzo Wada's color dyad. from ["A Dictionary of Color Combinations"](https://sanzo-wada.dmbk.io/)
use:
    - <ivec2> wadaDuads (<int> index)
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
ivec2 wadaDyad( const int index ) {
    ivec2 d[WADA_DYAD_TOTAL];
    d[0] = ivec2(65,115);
    d[1] = ivec2(13,39);
    d[2] = ivec2(50,148);
    d[3] = ivec2(7,129);
    d[4] = ivec2(63,87);
    d[5] = ivec2(66,131);
    d[6] = ivec2(36,138);
    d[7] = ivec2(66,72);
    d[8] = ivec2(41,153);
    d[9] = ivec2(50,117);
    d[10] = ivec2(13,143);
    d[11] = ivec2(9,40);
    d[12] = ivec2(98,131);
    d[13] = ivec2(33,109);
    d[14] = ivec2(12,97);
    d[15] = ivec2(84,91);
    d[16] = ivec2(113,133);
    d[17] = ivec2(7,97);
    d[18] = ivec2(82,143);
    d[19] = ivec2(18,108);
    d[20] = ivec2(47,81);
    d[21] = ivec2(1,155);
    d[22] = ivec2(83,129);
    d[23] = ivec2(74,111);
    d[24] = ivec2(29,153);
    d[25] = ivec2(17,39);
    d[26] = ivec2(49,91);
    d[27] = ivec2(13,155);
    d[28] = ivec2(8,152);
    d[29] = ivec2(4,28);
    d[30] = ivec2(68,86);
    d[31] = ivec2(27,148);
    d[32] = ivec2(103,124);
    d[33] = ivec2(22,121);
    d[34] = ivec2(72,113);
    d[35] = ivec2(59,147);
    d[36] = ivec2(1,137);
    d[37] = ivec2(99,118);
    d[38] = ivec2(42,55);
    d[39] = ivec2(63,156);
    d[40] = ivec2(18,132);
    d[41] = ivec2(24,121);
    d[42] = ivec2(109,119);
    d[43] = ivec2(41,150);
    d[44] = ivec2(28,119);
    d[45] = ivec2(38,156);
    d[46] = ivec2(61,150);
    d[47] = ivec2(98,110);
    d[48] = ivec2(10,151);
    d[49] = ivec2(132,147);
    d[50] = ivec2(145,155);
    d[51] = ivec2(32,97);
    d[52] = ivec2(8,70);
    d[53] = ivec2(39,127);
    d[54] = ivec2(90,149);
    d[55] = ivec2(54,156);
    d[56] = ivec2(37,115);
    d[57] = ivec2(137,139);
    d[58] = ivec2(68,113);
    d[59] = ivec2(58,78);
    d[60] = ivec2(118,127);
    d[61] = ivec2(4,54);
    d[62] = ivec2(154,156);
    d[63] = ivec2(13,105);
    d[64] = ivec2(29,49);
    d[65] = ivec2(38,109);
    d[66] = ivec2(86,117);
    d[67] = ivec2(109,125);
    d[68] = ivec2(39,154);
    d[69] = ivec2(44,114);
    d[70] = ivec2(83,117);
    d[71] = ivec2(47,154);
    d[72] = ivec2(32,150);
    d[73] = ivec2(52,125);
    d[74] = ivec2(42,130);
    d[75] = ivec2(14,120);
    d[76] = ivec2(13,97);
    d[77] = ivec2(42,119);
    d[78] = ivec2(61,125);
    d[79] = ivec2(8,137);
    d[80] = ivec2(37,67);
    d[81] = ivec2(5,98);
    d[82] = ivec2(70,110);
    d[83] = ivec2(41,123);
    d[84] = ivec2(32,128);
    d[85] = ivec2(59,75);
    d[86] = ivec2(1,18);
    d[87] = ivec2(83,125);
    d[88] = ivec2(39,115);
    d[89] = ivec2(71,138);
    d[90] = ivec2(41,67);
    d[91] = ivec2(66,150);
    d[92] = ivec2(28,38);
    d[93] = ivec2(120,128);
    d[94] = ivec2(56,73);
    d[95] = ivec2(8,27);
    d[96] = ivec2(39,106);
    d[97] = ivec2(31,85);
    d[98] = ivec2(39,89);
    d[99] = ivec2(7,156);
    d[100] = ivec2(42,85);
    d[101] = ivec2(60,120);
    d[102] = ivec2(40,64);
    d[103] = ivec2(22,156);
    d[104] = ivec2(59,85);
    d[105] = ivec2(110,127);
    d[106] = ivec2(18,45);
    d[107] = ivec2(31,128);
    d[108] = ivec2(42,87);
    d[109] = ivec2(38,61);
    d[110] = ivec2(8,60);
    d[111] = ivec2(31,121);
    d[112] = ivec2(10,134);
    d[113] = ivec2(9,37);
    d[114] = ivec2(53,109);
    d[115] = ivec2(55,143);
    d[116] = ivec2(111,112);
    d[117] = ivec2(109,140);
    d[118] = ivec2(1,66);
    d[119] = ivec2(46,119);
    d[120] = ivec2(107,152);
    d[121] = ivec2(32,118);
    d[122] = ivec2(62,125);
    d[123] = ivec2(7,38);

    #if defined(PLATFORM_WEBGL)
    for (int i = 0; i < WADA_DYAD_TOTAL; i++)
        if (i == index) return d[i];
    #else
    return d[index];    
    #endif
}
#endif
