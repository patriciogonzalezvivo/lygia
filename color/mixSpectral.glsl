#include "space/srgb2rgb.glsl"

/*
contributors: Ronald van Wijnen (@OneDayOfCrypto)
description: | 
    Spectral mix allows you to achieve realistic color mixing in your projects. 
    It is based on the Kubelka-Munk theory, a proven scientific model that simulates 
    how light interacts with paint to produce lifelike color mixing. 
    Find more informatiom on Ronald van Wijnen's [original repository](https://github.com/rvanwijnen/spectral.js)
options:
    - MIXSPECTRAL_SRGB: by default A and B are linear RGB. If you want to use sRGB, define this flag.
use: <vec3\vec4> mixSpectral(<vec3|vec4> A, <vec3|vec4> B, float pct)
examples:
    - /shaders/color_mix.frag
license: MIT License Copyright (c) 2023 Ronald van Wijnen
*/

#ifndef FNC_MIXSPECTRAL
#define FNC_MIXSPECTRAL

#define MIXSPECTRAL_SIZE 38
#define MIXSPECTRAL_EPSILON 0.0001

const mat3 XYZ2RGB = mat3(
     3.2409699419045200,-0.9692436362808790, 0.0556300796969936,
    -1.5373831775700900, 1.8759675015077200,-0.2039769588889760,
    -0.4986107602930030, 0.0415550574071756, 1.0569715142428700
);

void mixSpectral_linear_to_reflectance(vec3 lrgb, inout float R[MIXSPECTRAL_SIZE]) {
    float w = min(lrgb.r, min(lrgb.g, lrgb.b));
    lrgb -= w;

    float c = min(lrgb.g, lrgb.b);
    float m = min(lrgb.r, lrgb.b);
    float y = min(lrgb.r, lrgb.g);

    float r = min(max(0.0, lrgb.r - lrgb.b), max(0.0, lrgb.r - lrgb.g));
    float g = min(max(0.0, lrgb.g - lrgb.b), max(0.0, lrgb.g - lrgb.r));
    float b = min(max(0.0, lrgb.b - lrgb.g), max(0.0, lrgb.b - lrgb.r));
    
    R[0] = max(MIXSPECTRAL_EPSILON, w * 1.0011607271876400 + c * 0.9705850013229620 + m * 0.9906735573199880 + y * 0.0210523371789306 + r * 0.0315605737777207 + g * 0.0095560747554212 + b * 0.9794047525020140);
    R[1] = max(MIXSPECTRAL_EPSILON, w * 1.0011606515972800 + c * 0.9705924981434250 + m * 0.9906715249619790 + y * 0.0210564627517414 + r * 0.0315520718330149 + g * 0.0095581580120851 + b * 0.9794007068431300);
    R[2] = max(MIXSPECTRAL_EPSILON, w * 1.0011603192274700 + c * 0.9706253487298910 + m * 0.9906625823534210 + y * 0.0210746178695038 + r * 0.0315148215513658 + g * 0.0095673245444588 + b * 0.9793829034702610);
    R[3] = max(MIXSPECTRAL_EPSILON, w * 1.0011586727078900 + c * 0.9707868061190170 + m * 0.9906181076447950 + y * 0.0211649058448753 + r * 0.0313318044982702 + g * 0.0096129126297349 + b * 0.9792943649455940);
    R[4] = max(MIXSPECTRAL_EPSILON, w * 1.0011525984455200 + c * 0.9713686732282480 + m * 0.9904514808787100 + y * 0.0215027957272504 + r * 0.0306729857725527 + g * 0.0097837090401843 + b * 0.9789630146085700);
    R[5] = max(MIXSPECTRAL_EPSILON, w * 1.0011325252899800 + c * 0.9731632306212520 + m * 0.9898710814002040 + y * 0.0226738799041561 + r * 0.0286480476989607 + g * 0.0103786227058710 + b * 0.9778144666940430);
    R[6] = max(MIXSPECTRAL_EPSILON, w * 1.0010850066332700 + c * 0.9767402231587650 + m * 0.9882866087596400 + y * 0.0258235649693629 + r * 0.0246450407045709 + g * 0.0120026452378567 + b * 0.9747243211338360);
    R[7] = max(MIXSPECTRAL_EPSILON, w * 1.0009968788945300 + c * 0.9815876054913770 + m * 0.9842906927975040 + y * 0.0334879385639851 + r * 0.0192960753663651 + g * 0.0160977721473922 + b * 0.9671984823439730);
    R[8] = max(MIXSPECTRAL_EPSILON, w * 1.0008652515227400 + c * 0.9862802656529490 + m * 0.9739349056253060 + y * 0.0519069663740307 + r * 0.0142066612220556 + g * 0.0267061902231680 + b * 0.9490796575305750);
    R[9] = max(MIXSPECTRAL_EPSILON, w * 1.0006962900094000 + c * 0.9899491476891340 + m * 0.9418178384601450 + y * 0.1007490148334730 + r * 0.0102942608878609 + g * 0.0595555440185881 + b * 0.9008501289409770);
    R[10] = max(MIXSPECTRAL_EPSILON, w * 1.0005049611488800 + c * 0.9924927015384200 + m * 0.8173903261951560 + y * 0.2391298997068470 + r * 0.0076191460521811 + g * 0.1860398265328260 + b * 0.7631504454622400);
    R[11] = max(MIXSPECTRAL_EPSILON, w * 1.0003080818799200 + c * 0.9941456804052560 + m * 0.4324728050657290 + y * 0.5348043122727480 + r * 0.0058980410835420 + g * 0.5705798201161590 + b * 0.4659221716493190);
    R[12] = max(MIXSPECTRAL_EPSILON, w * 1.0001196660201300 + c * 0.9951839750332120 + m * 0.1384539782588700 + y * 0.7978075786430300 + r * 0.0048233247781713 + g * 0.8614677684002920 + b * 0.2012632804510050);
    R[13] = max(MIXSPECTRAL_EPSILON, w * 0.9999527659684070 + c * 0.9957567501108180 + m * 0.0537347216940033 + y * 0.9114498940673840 + r * 0.0042298748350633 + g * 0.9458790897676580 + b * 0.0877524413419623);
    R[14] = max(MIXSPECTRAL_EPSILON, w * 0.9998218368992970 + c * 0.9959128182867100 + m * 0.0292174996673231 + y * 0.9537979630045070 + r * 0.0040599171299341 + g * 0.9704654864743050 + b * 0.0457176793291679);
    R[15] = max(MIXSPECTRAL_EPSILON, w * 0.9997386095575930 + c * 0.9956061578345280 + m * 0.0213136517508590 + y * 0.9712416154654290 + r * 0.0043533695594676 + g * 0.9784136302844500 + b * 0.0284706050521843);
    R[16] = max(MIXSPECTRAL_EPSILON, w * 0.9997095516396120 + c * 0.9945976009618540 + m * 0.0201349530181136 + y * 0.9793031238075880 + r * 0.0053434425970201 + g * 0.9795890314112240 + b * 0.0205271767569850);
    R[17] = max(MIXSPECTRAL_EPSILON, w * 0.9997319302106270 + c * 0.9922157154923700 + m * 0.0241323096280662 + y * 0.9833801195075750 + r * 0.0076917201010463 + g * 0.9755335369086320 + b * 0.0165302792310211);
    R[18] = max(MIXSPECTRAL_EPSILON, w * 0.9997994363461950 + c * 0.9862364527832490 + m * 0.0372236145223627 + y * 0.9854612465677550 + r * 0.0135969795736536 + g * 0.9622887553978130 + b * 0.0145135107212858);
    R[19] = max(MIXSPECTRAL_EPSILON, w * 0.9999003303166710 + c * 0.9679433372645410 + m * 0.0760506552706601 + y * 0.9864350469766050 + r * 0.0316975442661115 + g * 0.9231215745131200 + b * 0.0136003508637687);
    R[20] = max(MIXSPECTRAL_EPSILON, w * 1.0000204065261100 + c * 0.8912850042449430 + m * 0.2053754719423990 + y * 0.9867382506701410 + r * 0.1078611963552490 + g * 0.7934340189431110 + b * 0.0133604258769571);
    R[21] = max(MIXSPECTRAL_EPSILON, w * 1.0001447879365800 + c * 0.5362024778620530 + m * 0.5412689034604390 + y * 0.9866178824450320 + r * 0.4638126031687040 + g * 0.4592701359024290 + b * 0.0135488943145680);
    R[22] = max(MIXSPECTRAL_EPSILON, w * 1.0002599790341200 + c * 0.1541081190018780 + m * 0.8158416850864860 + y * 0.9862777767586430 + r * 0.8470554052720110 + g * 0.1855741036663030 + b * 0.0139594356366992);
    R[23] = max(MIXSPECTRAL_EPSILON, w * 1.0003557969708900 + c * 0.0574575093228929 + m * 0.9128177041239760 + y * 0.9858605924440560 + r * 0.9431854093939180 + g * 0.0881774959955372 + b * 0.0144434255753570);
    R[24] = max(MIXSPECTRAL_EPSILON, w * 1.0004275378026900 + c * 0.0315349873107007 + m * 0.9463398301669620 + y * 0.9854749276762100 + r * 0.9688621506965580 + g * 0.0543630228766700 + b * 0.0148854440621406);
    R[25] = max(MIXSPECTRAL_EPSILON, w * 1.0004762334488800 + c * 0.0222633920086335 + m * 0.9599276963319910 + y * 0.9851769347655580 + r * 0.9780306674736030 + g * 0.0406288447060719 + b * 0.0152254296999746);
    R[26] = max(MIXSPECTRAL_EPSILON, w * 1.0005072096750800 + c * 0.0182022841492439 + m * 0.9662605952303120 + y * 0.9849715740141810 + r * 0.9820436438543060 + g * 0.0342215204316970 + b * 0.0154592848180209);
    R[27] = max(MIXSPECTRAL_EPSILON, w * 1.0005251915637300 + c * 0.0162990559732640 + m * 0.9693259700584240 + y * 0.9848463034157120 + r * 0.9839236237187070 + g * 0.0311185790956966 + b * 0.0156018026485961);
    R[28] = max(MIXSPECTRAL_EPSILON, w * 1.0005350960689600 + c * 0.0153656239334613 + m * 0.9708545367213990 + y * 0.9847753518111990 + r * 0.9848454841543820 + g * 0.0295708898336134 + b * 0.0156824871281936);
    R[29] = max(MIXSPECTRAL_EPSILON, w * 1.0005402209748200 + c * 0.0149111568733976 + m * 0.9716050665281280 + y * 0.9847380666252650 + r * 0.9852942758145960 + g * 0.0288108739348928 + b * 0.0157248764360615);
    R[30] = max(MIXSPECTRAL_EPSILON, w * 1.0005427281678400 + c * 0.0146954339898235 + m * 0.9719627697573920 + y * 0.9847196483117650 + r * 0.9855072952198250 + g * 0.0284486271324597 + b * 0.0157458108784121);
    R[31] = max(MIXSPECTRAL_EPSILON, w * 1.0005438956908700 + c * 0.0145964146717719 + m * 0.9721272722745090 + y * 0.9847110233919390 + r * 0.9856050715398370 + g * 0.0282820301724731 + b * 0.0157556123350225);
    R[32] = max(MIXSPECTRAL_EPSILON, w * 1.0005444821215100 + c * 0.0145470156699655 + m * 0.9722094177458120 + y * 0.9847066833006760 + r * 0.9856538499335780 + g * 0.0281988376490237 + b * 0.0157605443964911);
    R[33] = max(MIXSPECTRAL_EPSILON, w * 1.0005447695999200 + c * 0.0145228771899495 + m * 0.9722495776784240 + y * 0.9847045543930910 + r * 0.9856776850338830 + g * 0.0281581655342037 + b * 0.0157629637515278);
    R[34] = max(MIXSPECTRAL_EPSILON, w * 1.0005448988776200 + c * 0.0145120341118965 + m * 0.9722676219987420 + y * 0.9847035963093700 + r * 0.9856883918061220 + g * 0.0281398910216386 + b * 0.0157640525629106);
    R[35] = max(MIXSPECTRAL_EPSILON, w * 1.0005449625468900 + c * 0.0145066940939832 + m * 0.9722765094621500 + y * 0.9847031240775520 + r * 0.9856936646900310 + g * 0.0281308901665811 + b * 0.0157645892329510);
    R[36] = max(MIXSPECTRAL_EPSILON, w * 1.0005449892705800 + c * 0.0145044507314479 + m * 0.9722802433068740 + y * 0.9847029256150900 + r * 0.9856958798482050 + g * 0.0281271086805816 + b * 0.0157648147772649);
    R[37] = max(MIXSPECTRAL_EPSILON, w * 1.0005449969930000 + c * 0.0145038009464639 + m * 0.9722813248265600 + y * 0.9847028681227950 + r * 0.9856965214637620 + g * 0.0281260133612096 + b * 0.0157648801149616);
}

vec3 mixSpectral_reflectance_to_xyz(float R[MIXSPECTRAL_SIZE]) {
    return R[ 0] * vec3(0.0000646919989576, 0.0000018442894440, 0.0003050171476380) +
		   R[ 1] * vec3(0.0002194098998132, 0.0000062053235865, 0.0010368066663574) +
		   R[ 2] * vec3(0.0011205743509343, 0.0000310096046799, 0.0053131363323992) +
		   R[ 3] * vec3(0.0037666134117111, 0.0001047483849269, 0.0179543925899536) +
		   R[ 4] * vec3(0.0118805536037990, 0.0003536405299538, 0.0570775815345485) +
		   R[ 5] * vec3(0.0232864424191771, 0.0009514714056444, 0.1136516189362870) +
		   R[ 6] * vec3(0.0345594181969747, 0.0022822631748318, 0.1733587261835500) +
		   R[ 7] * vec3(0.0372237901162006, 0.0042073290434730, 0.1962065755586570) +
    	   R[ 8] * vec3(0.0324183761091486, 0.0066887983719014, 0.1860823707062960) +
    	   R[ 9] * vec3(0.0212332056093810, 0.0098883960193565, 0.1399504753832070) +
    	   R[10] * vec3(0.0104909907685421, 0.0152494514496311, 0.0891745294268649) +
    	   R[11] * vec3(0.0032958375797931, 0.0214183109449723, 0.0478962113517075) +
    	   R[12] * vec3(0.0005070351633801, 0.0334229301575068, 0.0281456253957952) +
    	   R[13] * vec3(0.0009486742057141, 0.0513100134918512, 0.0161376622950514) +
    	   R[14] * vec3(0.0062737180998318, 0.0704020839399490, 0.0077591019215214) +
    	   R[15] * vec3(0.0168646241897775, 0.0878387072603517, 0.0042961483736618) +
    	   R[16] * vec3(0.0286896490259810, 0.0942490536184085, 0.0020055092122156) +
    	   R[17] * vec3(0.0426748124691731, 0.0979566702718931, 0.0008614711098802) +
    	   R[18] * vec3(0.0562547481311377, 0.0941521856862608, 0.0003690387177652) +
    	   R[19] * vec3(0.0694703972677158, 0.0867810237486753, 0.0001914287288574) +
    	   R[20] * vec3(0.0830531516998291, 0.0788565338632013, 0.0001495555858975) +
    	   R[21] * vec3(0.0861260963002257, 0.0635267026203555, 0.0000923109285104) +
    	   R[22] * vec3(0.0904661376847769, 0.0537414167568200, 0.0000681349182337) +
    	   R[23] * vec3(0.0850038650591277, 0.0426460643574120, 0.0000288263655696) +
    	   R[24] * vec3(0.0709066691074488, 0.0316173492792708, 0.0000157671820553) +
    	   R[25] * vec3(0.0506288916373645, 0.0208852059213910, 0.0000039406041027) +
    	   R[26] * vec3(0.0354739618852640, 0.0138601101360152, 0.0000015840125870) +
    	   R[27] * vec3(0.0214682102597065, 0.0081026402038399, 0.0000000000000000) +
    	   R[28] * vec3(0.0125164567619117, 0.0046301022588030, 0.0000000000000000) +
    	   R[29] * vec3(0.0068045816390165, 0.0024913800051319, 0.0000000000000000) +
    	   R[30] * vec3(0.0034645657946526, 0.0012593033677378, 0.0000000000000000) +
    	   R[31] * vec3(0.0014976097506959, 0.0005416465221680, 0.0000000000000000) +
    	   R[32] * vec3(0.0007697004809280, 0.0002779528920067, 0.0000000000000000) +
    	   R[33] * vec3(0.0004073680581315, 0.0001471080673854, 0.0000000000000000) +
    	   R[34] * vec3(0.0001690104031614, 0.0000610327472927, 0.0000000000000000) +
    	   R[35] * vec3(0.0000952245150365, 0.0000343873229523, 0.0000000000000000) +
    	   R[36] * vec3(0.0000490309872958, 0.0000177059860053, 0.0000000000000000) +
    	   R[37] * vec3(0.0000199961492222, 0.0000072209749130, 0.0000000000000000);
}

float KS(float R) {
	return pow(1.0 - R, 2.0) / (2.0 * R);
}

float KM(float KS) {
    return 1.0 + KS - sqrt(pow(KS, 2.0) + 2.0 * KS);
}

vec3 mixSpectral2(vec3 A, float t1, vec3 B, float t2) {
    #ifdef MIXSPECTRAL_SRGB
    vec3 lrgb1 = srgb2rgb(A);
    vec3 lrgb2 = srgb2rgb(B);
    #else
    vec3 lrgb1 = A;
    vec3 lrgb2 = B;
    #endif

    float R1[MIXSPECTRAL_SIZE];
    float R2[MIXSPECTRAL_SIZE];

    for (int i = 0; i < MIXSPECTRAL_SIZE; i++) {
        R1[i] = 0.0;
        R2[i] = 0.0;
    }

    mixSpectral_linear_to_reflectance(lrgb1, R1);
    mixSpectral_linear_to_reflectance(lrgb2, R2);

    float l1 = mixSpectral_reflectance_to_xyz(R1)[1];
    float l2 = mixSpectral_reflectance_to_xyz(R2)[1];

    float R[MIXSPECTRAL_SIZE];

    for (int i = 0; i < MIXSPECTRAL_SIZE; i++) {
		float concentration1 = pow(pow(l1, 0.5) * t1, 2.0);
		float concentration2 = pow(pow(l2, 0.5) * t2, 2.0);
		
		float totalConcentration = concentration1 + concentration2;
		
        float ksMix = 0.0;
		
		ksMix += KS(R1[i]) * concentration1;
		ksMix += KS(R2[i]) * concentration2;

        R[i] = KM(ksMix / totalConcentration);
    }

    vec3 rgb = XYZ2RGB * mixSpectral_reflectance_to_xyz(R);

    #ifdef MIXSPECTRAL_SRGB
    return rgb2srgb(rgb);
    #else
    return rgb;
    #endif
}

vec3 mixSpectral(vec3 A, vec3 B, float t) {
	return mixSpectral2(A, (1.0 - t), B, t);
}

vec3 mixSpectral3(vec3 A, float t1, vec3 B, float t2, vec3 C, float t3) {
    #ifdef MIXSPECTRAL_SRGB
    vec3 lrgb1 = srgb2rgb(A);
    vec3 lrgb2 = srgb2rgb(B);
    vec3 lrgb3 = srgb2rgb(C);
    #else
    vec3 lrgb1 = A;
    vec3 lrgb2 = B;
    vec3 lrgb3 = C;
    #endif

    float R1[MIXSPECTRAL_SIZE];
    float R2[MIXSPECTRAL_SIZE];
    float R3[MIXSPECTRAL_SIZE];

    for (int i = 0; i < MIXSPECTRAL_SIZE; i++) {
        R1[i] = 0.0;
        R2[i] = 0.0;
        R3[i] = 0.0;
    }

    mixSpectral_linear_to_reflectance(lrgb1, R1);
    mixSpectral_linear_to_reflectance(lrgb2, R2);
    mixSpectral_linear_to_reflectance(lrgb3, R3);

    float l1 = mixSpectral_reflectance_to_xyz(R1)[1];
    float l2 = mixSpectral_reflectance_to_xyz(R2)[1];
    float l3 = mixSpectral_reflectance_to_xyz(R3)[1];

    float R[MIXSPECTRAL_SIZE];

    for (int i = 0; i < MIXSPECTRAL_SIZE; i++) {
		float concentration1 = pow(pow(l1, 0.5) * t1, 2.0);
		float concentration2 = pow(pow(l2, 0.5) * t2, 2.0);
		float concentration3 = pow(pow(l3, 0.5) * t3, 2.0);
		
		float totalConcentration = concentration1 + concentration2 + concentration3;

        float ksMix = 0.0;
		
		ksMix += KS(R1[i]) * concentration1;
		ksMix += KS(R2[i]) * concentration2;
		ksMix += KS(R3[i]) * concentration3;

        R[i] = KM(ksMix / totalConcentration);
    }

    vec3 rgb = XYZ2RGB * mixSpectral_reflectance_to_xyz(R);

    #ifdef MIXSPECTRAL_SRGB
    return rgb2srgb(rgb);
    #else
    return rgb;
    #endif
}

vec3 mixSpectral4(vec3 A, float t1, vec3 B, float t2, vec3 C, float t3, vec3 D, float t4) {
    #ifdef MIXSPECTRAL_SRGB
    vec3 lrgb1 = srgb2rgb(A);
    vec3 lrgb2 = srgb2rgb(B);
    vec3 lrgb3 = srgb2rgb(C);
    vec3 lrgb4 = srgb2rgb(D);
    #else
    vec3 lrgb1 = A;
    vec3 lrgb2 = B;
    vec3 lrgb3 = C;
    vec3 lrgb4 = D;
    #endif

    float R1[MIXSPECTRAL_SIZE];
    float R2[MIXSPECTRAL_SIZE];
    float R3[MIXSPECTRAL_SIZE];
    float R4[MIXSPECTRAL_SIZE];

    for (int i = 0; i < MIXSPECTRAL_SIZE; i++) {
        R1[i] = 0.0;
        R2[i] = 0.0;
        R3[i] = 0.0;
        R4[i] = 0.0;
    }

    mixSpectral_linear_to_reflectance(lrgb1, R1);
    mixSpectral_linear_to_reflectance(lrgb2, R2);
    mixSpectral_linear_to_reflectance(lrgb3, R3);
    mixSpectral_linear_to_reflectance(lrgb4, R4);

    float l1 = mixSpectral_reflectance_to_xyz(R1)[1];
    float l2 = mixSpectral_reflectance_to_xyz(R2)[1];
    float l3 = mixSpectral_reflectance_to_xyz(R3)[1];
    float l4 = mixSpectral_reflectance_to_xyz(R4)[1];

    float R[MIXSPECTRAL_SIZE];

    for (int i = 0; i < MIXSPECTRAL_SIZE; i++) {
		float concentration1 = pow(pow(l1, 0.5) * t1, 2.0);
		float concentration2 = pow(pow(l2, 0.5) * t2, 2.0);
		float concentration3 = pow(pow(l3, 0.5) * t3, 2.0);
		float concentration4 = pow(pow(l4, 0.5) * t4, 2.0);
		
		float totalConcentration = concentration1 + concentration2 + concentration3 + concentration4;

        float ksMix = 0.0;
		
		ksMix += KS(R1[i]) * concentration1;
		ksMix += KS(R2[i]) * concentration2;
		ksMix += KS(R3[i]) * concentration3;
		ksMix += KS(R4[i]) * concentration4;

        R[i] = KM(ksMix / totalConcentration);
    }

    vec3 rgb = XYZ2RGB * mixSpectral_reflectance_to_xyz(R);

    #ifdef MIXSPECTRAL_SRGB
    return rgb2srgb(rgb);
    #else
    return rgb;
    #endif
}

vec4 mixSpectral(vec4 A, vec4 B, float t) {
    return vec4(mixSpectral(A.rgb, B.rgb, t), mix(A.a, B.a, t));
}

#endif
