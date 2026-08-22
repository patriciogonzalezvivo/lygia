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

fn wadaDyad(index: i32) -> vec2i {
    const WADA_DYAD_TOTAL: f32 = 124;
    ivec2 d[WADA_DYAD_TOTAL];
    d[0] = vec2i(65,115);
    d[1] = vec2i(13,39);
    d[2] = vec2i(50,148);
    d[3] = vec2i(7,129);
    d[4] = vec2i(63,87);
    d[5] = vec2i(66,131);
    d[6] = vec2i(36,138);
    d[7] = vec2i(66,72);
    d[8] = vec2i(41,153);
    d[9] = vec2i(50,117);
    d[10] = vec2i(13,143);
    d[11] = vec2i(9,40);
    d[12] = vec2i(98,131);
    d[13] = vec2i(33,109);
    d[14] = vec2i(12,97);
    d[15] = vec2i(84,91);
    d[16] = vec2i(113,133);
    d[17] = vec2i(7,97);
    d[18] = vec2i(82,143);
    d[19] = vec2i(18,108);
    d[20] = vec2i(47,81);
    d[21] = vec2i(1,155);
    d[22] = vec2i(83,129);
    d[23] = vec2i(74,111);
    d[24] = vec2i(29,153);
    d[25] = vec2i(17,39);
    d[26] = vec2i(49,91);
    d[27] = vec2i(13,155);
    d[28] = vec2i(8,152);
    d[29] = vec2i(4,28);
    d[30] = vec2i(68,86);
    d[31] = vec2i(27,148);
    d[32] = vec2i(103,124);
    d[33] = vec2i(22,121);
    d[34] = vec2i(72,113);
    d[35] = vec2i(59,147);
    d[36] = vec2i(1,137);
    d[37] = vec2i(99,118);
    d[38] = vec2i(42,55);
    d[39] = vec2i(63,156);
    d[40] = vec2i(18,132);
    d[41] = vec2i(24,121);
    d[42] = vec2i(109,119);
    d[43] = vec2i(41,150);
    d[44] = vec2i(28,119);
    d[45] = vec2i(38,156);
    d[46] = vec2i(61,150);
    d[47] = vec2i(98,110);
    d[48] = vec2i(10,151);
    d[49] = vec2i(132,147);
    d[50] = vec2i(145,155);
    d[51] = vec2i(32,97);
    d[52] = vec2i(8,70);
    d[53] = vec2i(39,127);
    d[54] = vec2i(90,149);
    d[55] = vec2i(54,156);
    d[56] = vec2i(37,115);
    d[57] = vec2i(137,139);
    d[58] = vec2i(68,113);
    d[59] = vec2i(58,78);
    d[60] = vec2i(118,127);
    d[61] = vec2i(4,54);
    d[62] = vec2i(154,156);
    d[63] = vec2i(13,105);
    d[64] = vec2i(29,49);
    d[65] = vec2i(38,109);
    d[66] = vec2i(86,117);
    d[67] = vec2i(109,125);
    d[68] = vec2i(39,154);
    d[69] = vec2i(44,114);
    d[70] = vec2i(83,117);
    d[71] = vec2i(47,154);
    d[72] = vec2i(32,150);
    d[73] = vec2i(52,125);
    d[74] = vec2i(42,130);
    d[75] = vec2i(14,120);
    d[76] = vec2i(13,97);
    d[77] = vec2i(42,119);
    d[78] = vec2i(61,125);
    d[79] = vec2i(8,137);
    d[80] = vec2i(37,67);
    d[81] = vec2i(5,98);
    d[82] = vec2i(70,110);
    d[83] = vec2i(41,123);
    d[84] = vec2i(32,128);
    d[85] = vec2i(59,75);
    d[86] = vec2i(1,18);
    d[87] = vec2i(83,125);
    d[88] = vec2i(39,115);
    d[89] = vec2i(71,138);
    d[90] = vec2i(41,67);
    d[91] = vec2i(66,150);
    d[92] = vec2i(28,38);
    d[93] = vec2i(120,128);
    d[94] = vec2i(56,73);
    d[95] = vec2i(8,27);
    d[96] = vec2i(39,106);
    d[97] = vec2i(31,85);
    d[98] = vec2i(39,89);
    d[99] = vec2i(7,156);
    d[100] = vec2i(42,85);
    d[101] = vec2i(60,120);
    d[102] = vec2i(40,64);
    d[103] = vec2i(22,156);
    d[104] = vec2i(59,85);
    d[105] = vec2i(110,127);
    d[106] = vec2i(18,45);
    d[107] = vec2i(31,128);
    d[108] = vec2i(42,87);
    d[109] = vec2i(38,61);
    d[110] = vec2i(8,60);
    d[111] = vec2i(31,121);
    d[112] = vec2i(10,134);
    d[113] = vec2i(9,37);
    d[114] = vec2i(53,109);
    d[115] = vec2i(55,143);
    d[116] = vec2i(111,112);
    d[117] = vec2i(109,140);
    d[118] = vec2i(1,66);
    d[119] = vec2i(46,119);
    d[120] = vec2i(107,152);
    d[121] = vec2i(32,118);
    d[122] = vec2i(62,125);
    d[123] = vec2i(7,38);

    for (int i = 0; i < WADA_DYAD_TOTAL; i++)
        if (i == index) return d[i];
    return d[index];    
}
