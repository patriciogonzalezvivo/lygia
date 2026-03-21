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

const WADA_TETRAD_TOTAL: f32 = 99;

                
fn wadaTetrad(index: i32) -> vec4i {
    ivec4 t[WADA_TETRAD_TOTAL];
    t[0] = vec4i(2,3,115,125);
    t[1] = vec4i(1,114,135,136);
    t[2] = vec4i(8,135,136,148);
    t[3] = vec4i(45,135,136,152);
    t[4] = vec4i(19,56,100,156);
    t[5] = vec4i(28,83,96,111);
    t[6] = vec4i(17,39,50,122);
    t[7] = vec4i(8,19,103,156);
    t[8] = vec4i(13,41,78,155);
    t[9] = vec4i(4,14,94,120);
    t[10] = vec4i(28,102,127,155);
    t[11] = vec4i(1,27,38,45);
    t[12] = vec4i(13,56,98,124);
    t[13] = vec4i(8,69,131,144);
    t[14] = vec4i(32,53,58,122);
    t[15] = vec4i(57,64,97,108);
    t[16] = vec4i(30,54,103,155);
    t[17] = vec4i(12,13,68,117);
    t[18] = vec4i(55,62,149,155);
    t[19] = vec4i(1,38,75,133);
    t[20] = vec4i(13,57,113,156);
    t[21] = vec4i(48,63,101,156);
    t[22] = vec4i(16,60,119,137);
    t[23] = vec4i(35,44,75,120);
    t[24] = vec4i(55,117,121,154);
    t[25] = vec4i(10,48,87,97);
    t[26] = vec4i(30,39,113,154);
    t[27] = vec4i(12,41,70,104);
    t[28] = vec4i(19,44,86,155);
    t[29] = vec4i(1,17,88,115);
    t[30] = vec4i(10,56,92,128);
    t[31] = vec4i(46,61,98,119);
    t[32] = vec4i(4,13,107,108);
    t[33] = vec4i(13,35,137,156);
    t[34] = vec4i(12,38,78,104);
    t[35] = vec4i(25,112,117,130);
    t[36] = vec4i(39,63,86,111);
    t[37] = vec4i(0,34,77,152);
    t[38] = vec4i(18,53,83,145);
    t[39] = vec4i(8,42,89,156);
    t[40] = vec4i(9,24,75,128);
    t[41] = vec4i(46,58,104,123);
    t[42] = vec4i(13,48,53,127);
    t[43] = vec4i(12,105,133,145);
    t[44] = vec4i(39,98,112,120);
    t[45] = vec4i(12,51,112,134);
    t[46] = vec4i(20,35,95,114);
    t[47] = vec4i(11,56,97,123);
    t[48] = vec4i(4,19,64,86);
    t[49] = vec4i(19,60,116,125);
    t[50] = vec4i(8,57,109,113);
    t[51] = vec4i(61,82,145,156);
    t[52] = vec4i(55,90,125,128);
    t[53] = vec4i(37,39,105,112);
    t[54] = vec4i(90,97,112,113);
    t[55] = vec4i(39,44,50,53);
    t[56] = vec4i(13,86,93,100);
    t[57] = vec4i(38,46,104,111);
    t[58] = vec4i(46,54,119,128);
    t[59] = vec4i(38,49,81,155);
    t[60] = vec4i(19,61,78,125);
    t[61] = vec4i(23,48,102,120);
    t[62] = vec4i(7,46,86,113);
    t[63] = vec4i(46,53,108,120);
    t[64] = vec4i(40,64,107,152);
    t[65] = vec4i(32,46,72,98);
    t[66] = vec4i(55,98,101,108);
    t[67] = vec4i(22,131,137,149);
    t[68] = vec4i(38,44,75,107);
    t[69] = vec4i(29,46,88,90);
    t[70] = vec4i(19,61,93,121);
    t[71] = vec4i(22,54,103,156);
    t[72] = vec4i(8,32,124,150);
    t[73] = vec4i(33,101,147,148);
    t[74] = vec4i(43,53,55,86);
    t[75] = vec4i(79,106,123,130);
    t[76] = vec4i(13,56,61,104);
    t[77] = vec4i(5,38,102,110);
    t[78] = vec4i(4,38,107,111);
    t[79] = vec4i(29,118,137,152);
    t[80] = vec4i(12,40,59,130);
    t[81] = vec4i(38,64,89,91);
    t[82] = vec4i(8,13,53,132);
    t[83] = vec4i(46,132,149,155);
    t[84] = vec4i(52,108,111,117);
    t[85] = vec4i(5,21,107,123);
    t[86] = vec4i(19,55,112,119);
    t[87] = vec4i(42,75,89,120);
    t[88] = vec4i(33,61,147,155);
    t[89] = vec4i(8,32,39,106);
    t[90] = vec4i(37,133,146,156);
    t[91] = vec4i(49,65,110,125);
    t[92] = vec4i(64,97,152,156);
    t[93] = vec4i(7,104,107,110);
    t[94] = vec4i(1,46,76,107);
    t[95] = vec4i(45,124,137,156);
    t[96] = vec4i(32,108,114,140);
    t[97] = vec4i(92,97,121,134);
    t[98] = vec4i(52,104,107,149);

    for (int i = 0; i < WADA_TETRAD_TOTAL; i++)
        if (i == index) return t[i];
    return t[index];
}
