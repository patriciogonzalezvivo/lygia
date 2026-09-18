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

fn wadaTriad(index: i32) -> vec3i {
    const WADA_TRIAD_TOTAL: f32 = 112;
    ivec3 t[WADA_TRIAD_TOTAL];
    t[0] = vec3i(2,3,150);
    t[1] = vec3i(54,61,124);
    t[2] = vec3i(45,135,136);
    t[3] = vec3i(38,135,136);
    t[4] = vec3i(31,49,105);
    t[5] = vec3i(22,46,98);
    t[6] = vec3i(5,55,145);
    t[7] = vec3i(35,59,92);
    t[8] = vec3i(41,59,124);
    t[9] = vec3i(45,96,139);
    t[10] = vec3i(56,69,111);
    t[11] = vec3i(13,28,147);
    t[12] = vec3i(13,65,116);
    t[13] = vec3i(38,47,70);
    t[14] = vec3i(33,70,97);
    t[15] = vec3i(38,104,111);
    t[16] = vec3i(21,101,148);
    t[17] = vec3i(47,55,114);
    t[18] = vec3i(111,129,152);
    t[19] = vec3i(47,120,155);
    t[20] = vec3i(63,89,127);
    t[21] = vec3i(26,68,111);
    t[22] = vec3i(119,134,154);
    t[23] = vec3i(24,63,156);
    t[24] = vec3i(69,80,103);
    t[25] = vec3i(9,33,86);
    t[26] = vec3i(58,60,115);
    t[27] = vec3i(58,63,130);
    t[28] = vec3i(18,32,110);
    t[29] = vec3i(22,54,119);
    t[30] = vec3i(15,98,129);
    t[31] = vec3i(58,112,147);
    t[32] = vec3i(34,58,118);
    t[33] = vec3i(55,66,91);
    t[34] = vec3i(69,113,131);
    t[35] = vec3i(68,81,122);
    t[36] = vec3i(56,86,120);
    t[37] = vec3i(17,60,147);
    t[38] = vec3i(7,40,130);
    t[39] = vec3i(1,39,154);
    t[40] = vec3i(24,60,148);
    t[41] = vec3i(35,61,87);
    t[42] = vec3i(66,120,148);
    t[43] = vec3i(55,83,86);
    t[44] = vec3i(1,132,145);
    t[45] = vec3i(44,52,140);
    t[46] = vec3i(0,42,113);
    t[47] = vec3i(35,71,131);
    t[48] = vec3i(41,110,117);
    t[49] = vec3i(17,47,124);
    t[50] = vec3i(28,147,149);
    t[51] = vec3i(13,85,129);
    t[52] = vec3i(142,143,148);
    t[53] = vec3i(9,41,73);
    t[54] = vec3i(4,18,39);
    t[55] = vec3i(32,49,119);
    t[56] = vec3i(121,132,137);
    t[57] = vec3i(55,107,114);
    t[58] = vec3i(41,65,156);
    t[59] = vec3i(4,59,119);
    t[60] = vec3i(46,85,142);
    t[61] = vec3i(7,40,99);
    t[62] = vec3i(15,42,118);
    t[63] = vec3i(9,39,152);
    t[64] = vec3i(8,139,152);
    t[65] = vec3i(20,79,124);
    t[66] = vec3i(28,52,95);
    t[67] = vec3i(7,75,112);
    t[68] = vec3i(86,112,155);
    t[69] = vec3i(39,48,105);
    t[70] = vec3i(24,66,110);
    t[71] = vec3i(35,48,147);
    t[72] = vec3i(77,87,156);
    t[73] = vec3i(38,117,120);
    t[74] = vec3i(41,61,111);
    t[75] = vec3i(45,55,105);
    t[76] = vec3i(62,92,129);
    t[77] = vec3i(29,76,111);
    t[78] = vec3i(48,56,109);
    t[79] = vec3i(41,77,147);
    t[80] = vec3i(15,100,156);
    t[81] = vec3i(35,49,103);
    t[82] = vec3i(121,131,142);
    t[83] = vec3i(15,95,123);
    t[84] = vec3i(25,49,137);
    t[85] = vec3i(28,152,156);
    t[86] = vec3i(59,61,67);
    t[87] = vec3i(4,49,86);
    t[88] = vec3i(9,122,145);
    t[89] = vec3i(22,123,144);
    t[90] = vec3i(37,46,147);
    t[91] = vec3i(0,110,115);
    t[92] = vec3i(28,39,152);
    t[93] = vec3i(7,86,112);
    t[94] = vec3i(22,44,129);
    t[95] = vec3i(28,71,125);
    t[96] = vec3i(45,81,109);
    t[97] = vec3i(41,61,132);
    t[98] = vec3i(69,124,141);
    t[99] = vec3i(49,149,154);
    t[100] = vec3i(4,57,87);
    t[101] = vec3i(6,54,115);
    t[102] = vec3i(16,41,98);
    t[103] = vec3i(13,55,64);
    t[104] = vec3i(16,41,137);
    t[105] = vec3i(44,56,86);
    t[106] = vec3i(16,27,140);
    t[107] = vec3i(45,83,156);
    t[108] = vec3i(27,62,99);
    t[109] = vec3i(60,123,132);
    t[110] = vec3i(19,41,80);
    t[111] = vec3i(86,90,94);

    for (int i = 0; i < WADA_TRIAD_TOTAL; i++)
        if (i == index) return t[i];
    return t[index];
}
