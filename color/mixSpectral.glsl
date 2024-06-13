#include "space/srgb2rgb.glsl"
#include "space/rgb2srgb.glsl"
#include "space/xyz2srgb.glsl"

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

void mixSpectral_linear_to_reflectance(vec3 lrgb, inout float R[MIXSPECTRAL_SIZE]) {
    float w = min(lrgb.r, min(lrgb.g, lrgb.b));
    lrgb -= w;

    float c = min(lrgb.g, lrgb.b);
    float m = min(lrgb.r, lrgb.b);
    float y = min(lrgb.r, lrgb.g);

    float r = min(max(0.0, lrgb.r - lrgb.b), max(0.0, lrgb.r - lrgb.g));
    float g = min(max(0.0, lrgb.g - lrgb.b), max(0.0, lrgb.g - lrgb.r));
    float b = min(max(0.0, lrgb.b - lrgb.g), max(0.0, lrgb.b - lrgb.r));
    
    R[ 0] = max(MIXSPECTRAL_EPSILON, w + c * 0.96853629 + m * 0.51567122 + y * 0.02055257 + r * 0.03147571 + g * 0.49108579 + b * 0.97901834);
    R[ 1] = max(MIXSPECTRAL_EPSILON, w + c * 0.96855103 + m * 0.54015520 + y * 0.02059936 + r * 0.03146636 + g * 0.46944057 + b * 0.97901649);
    R[ 2] = max(MIXSPECTRAL_EPSILON, w + c * 0.96859338 + m * 0.62645502 + y * 0.02062723 + r * 0.03140624 + g * 0.40165780 + b * 0.97901118);
    R[ 3] = max(MIXSPECTRAL_EPSILON, w + c * 0.96877345 + m * 0.75595012 + y * 0.02073387 + r * 0.03119611 + g * 0.24490420 + b * 0.97892146);
    R[ 4] = max(MIXSPECTRAL_EPSILON, w + c * 0.96942204 + m * 0.92826996 + y * 0.02114202 + r * 0.03053888 + g * 0.06826880 + b * 0.97858555);
    R[ 5] = max(MIXSPECTRAL_EPSILON, w + c * 0.97143709 + m * 0.97223624 + y * 0.02233154 + r * 0.02856855 + g * 0.02732883 + b * 0.97743705);
    R[ 6] = max(MIXSPECTRAL_EPSILON, w + c * 0.97541862 + m * 0.98616174 + y * 0.02556857 + r * 0.02459485 + g * 0.01360600 + b * 0.97428075);
    R[ 7] = max(MIXSPECTRAL_EPSILON, w + c * 0.98074186 + m * 0.98955255 + y * 0.03330189 + r * 0.01929520 + g * 0.01000187 + b * 0.96663223);
    R[ 8] = max(MIXSPECTRAL_EPSILON, w + c * 0.98580992 + m * 0.98676237 + y * 0.05185294 + r * 0.01423112 + g * 0.01284127 + b * 0.94822893);
    R[ 9] = max(MIXSPECTRAL_EPSILON, w + c * 0.98971194 + m * 0.97312575 + y * 0.10087639 + r * 0.01033111 + g * 0.02636635 + b * 0.89937713);
    R[10] = max(MIXSPECTRAL_EPSILON, w + c * 0.99238027 + m * 0.91944277 + y * 0.24000413 + r * 0.00765876 + g * 0.07058713 + b * 0.76070164);
    R[11] = max(MIXSPECTRAL_EPSILON, w + c * 0.99409844 + m * 0.32564851 + y * 0.53589066 + r * 0.00593693 + g * 0.70421692 + b * 0.46420440);
    R[12] = max(MIXSPECTRAL_EPSILON, w + c * 0.99517200 + m * 0.13820628 + y * 0.79874659 + r * 0.00485616 + g * 0.85473994 + b * 0.20123039);
    R[13] = max(MIXSPECTRAL_EPSILON, w + c * 0.99576545 + m * 0.05015143 + y * 0.91186529 + r * 0.00426186 + g * 0.95081565 + b * 0.08808402);
    R[14] = max(MIXSPECTRAL_EPSILON, w + c * 0.99593552 + m * 0.02912336 + y * 0.95399623 + r * 0.00409039 + g * 0.97170370 + b * 0.04592894);
    R[15] = max(MIXSPECTRAL_EPSILON, w + c * 0.99564041 + m * 0.02421691 + y * 0.97137099 + r * 0.00438375 + g * 0.97651888 + b * 0.02860373);
    R[16] = max(MIXSPECTRAL_EPSILON, w + c * 0.99464769 + m * 0.02660696 + y * 0.97939505 + r * 0.00537525 + g * 0.97429245 + b * 0.02060067);
    R[17] = max(MIXSPECTRAL_EPSILON, w + c * 0.99229579 + m * 0.03407586 + y * 0.98345207 + r * 0.00772962 + g * 0.97012917 + b * 0.01656701);
    R[18] = max(MIXSPECTRAL_EPSILON, w + c * 0.98638762 + m * 0.04835936 + y * 0.98553736 + r * 0.01366120 + g * 0.94258630 + b * 0.01451549);
    R[19] = max(MIXSPECTRAL_EPSILON, w + c * 0.96829712 + m * 0.00011720 + y * 0.98648905 + r * 0.03181352 + g * 0.99989207 + b * 0.01357964);
    R[20] = max(MIXSPECTRAL_EPSILON, w + c * 0.89228016 + m * 0.00008554 + y * 0.98674535 + r * 0.10791525 + g * 0.99989891 + b * 0.01331243);
    R[21] = max(MIXSPECTRAL_EPSILON, w + c * 0.53740239 + m * 0.85267882 + y * 0.98657555 + r * 0.46249516 + g * 0.13823139 + b * 0.01347661);
    R[22] = max(MIXSPECTRAL_EPSILON, w + c * 0.15360445 + m * 0.93188793 + y * 0.98611877 + r * 0.84604333 + g * 0.06968113 + b * 0.01387181);
    R[23] = max(MIXSPECTRAL_EPSILON, w + c * 0.05705719 + m * 0.94810268 + y * 0.98559942 + r * 0.94275572 + g * 0.05628787 + b * 0.01435472);
    R[24] = max(MIXSPECTRAL_EPSILON, w + c * 0.03126539 + m * 0.94200977 + y * 0.98507063 + r * 0.96860996 + g * 0.06111561 + b * 0.01479836);
    R[25] = max(MIXSPECTRAL_EPSILON, w + c * 0.02205445 + m * 0.91478045 + y * 0.98460039 + r * 0.97783966 + g * 0.08987709 + b * 0.01515250);
    R[26] = max(MIXSPECTRAL_EPSILON, w + c * 0.01802271 + m * 0.87065445 + y * 0.98425301 + r * 0.98187757 + g * 0.13656016 + b * 0.01540513);
    R[27] = max(MIXSPECTRAL_EPSILON, w + c * 0.01613460 + m * 0.78827548 + y * 0.98403909 + r * 0.98377315 + g * 0.22169624 + b * 0.01557233);
    R[28] = max(MIXSPECTRAL_EPSILON, w + c * 0.01520947 + m * 0.65738359 + y * 0.98388535 + r * 0.98470202 + g * 0.32176956 + b * 0.01565710);
    R[29] = max(MIXSPECTRAL_EPSILON, w + c * 0.01475977 + m * 0.59909403 + y * 0.98376116 + r * 0.98515481 + g * 0.36157329 + b * 0.01571025);
    R[30] = max(MIXSPECTRAL_EPSILON, w + c * 0.01454263 + m * 0.56817268 + y * 0.98368246 + r * 0.98537114 + g * 0.48361920 + b * 0.01571916);
    R[31] = max(MIXSPECTRAL_EPSILON, w + c * 0.01444459 + m * 0.54031997 + y * 0.98365023 + r * 0.98546685 + g * 0.46488579 + b * 0.01572133);
    R[32] = max(MIXSPECTRAL_EPSILON, w + c * 0.01439897 + m * 0.52110241 + y * 0.98361309 + r * 0.98550011 + g * 0.47440306 + b * 0.01572502);
    R[33] = max(MIXSPECTRAL_EPSILON, w + c * 0.01437620 + m * 0.51041094 + y * 0.98357259 + r * 0.98551031 + g * 0.48576990 + b * 0.01571717);
    R[34] = max(MIXSPECTRAL_EPSILON, w + c * 0.01436343 + m * 0.50526577 + y * 0.98353856 + r * 0.98550741 + g * 0.49267971 + b * 0.01571905);
    R[35] = max(MIXSPECTRAL_EPSILON, w + c * 0.01435687 + m * 0.50255080 + y * 0.98351247 + r * 0.98551323 + g * 0.49625685 + b * 0.01571059);
    R[36] = max(MIXSPECTRAL_EPSILON, w + c * 0.01435370 + m * 0.50126452 + y * 0.98350101 + r * 0.98551563 + g * 0.49807754 + b * 0.01569728);
    R[37] = max(MIXSPECTRAL_EPSILON, w + c * 0.01435408 + m * 0.50083021 + y * 0.98350852 + r * 0.98551547 + g * 0.49889859 + b * 0.01570020);
}

vec3 mixSpectral_reflectance_to_xyz(float R[MIXSPECTRAL_SIZE]) {
    return  R[ 0] * vec3(0.00006469, 0.00000184, 0.00030502) +
            R[ 1] * vec3(0.00021941, 0.00000621, 0.00103681) +
            R[ 2] * vec3(0.00112057, 0.00003101, 0.00531314) +
            R[ 3] * vec3(0.00376661, 0.00010475, 0.01795439) +
            R[ 4] * vec3(0.01188055, 0.00035364, 0.05707758) +
            R[ 5] * vec3(0.02328644, 0.00095147, 0.11365162) +
            R[ 6] * vec3(0.03455942, 0.00228226, 0.17335873) +
            R[ 7] * vec3(0.03722379, 0.00420733, 0.19620658) +
            R[ 8] * vec3(0.03241838, 0.00668880, 0.18608237) +
            R[ 9] * vec3(0.02123321, 0.00988840, 0.13995048) +
            R[10] * vec3(0.01049099, 0.01524945, 0.08917453) +
            R[11] * vec3(0.00329584, 0.02141831, 0.04789621) +
            R[12] * vec3(0.00050704, 0.03342293, 0.02814563) +
            R[13] * vec3(0.00094867, 0.05131001, 0.01613766) +
            R[14] * vec3(0.00627372, 0.07040208, 0.00775910) +
            R[15] * vec3(0.01686462, 0.08783871, 0.00429615) +
            R[16] * vec3(0.02868965, 0.09424905, 0.00200551) +
            R[17] * vec3(0.04267481, 0.09795667, 0.00086147) +
            R[18] * vec3(0.05625475, 0.09415219, 0.00036904) +
            R[19] * vec3(0.06947040, 0.08678102, 0.00019143) +
            R[20] * vec3(0.08305315, 0.07885653, 0.00014956) +
            R[21] * vec3(0.08612610, 0.06352670, 0.00009231) +
            R[22] * vec3(0.09046614, 0.05374142, 0.00006813) +
            R[23] * vec3(0.08500387, 0.04264606, 0.00002883) +
            R[24] * vec3(0.07090667, 0.03161735, 0.00001577) +
            R[25] * vec3(0.05062889, 0.02088521, 0.00000394) +
            R[26] * vec3(0.03547396, 0.01386011, 0.00000158) +
            R[27] * vec3(0.02146821, 0.00810264, 0.00000000) +
            R[28] * vec3(0.01251646, 0.00463010, 0.00000000) +
            R[29] * vec3(0.00680458, 0.00249138, 0.00000000) +
            R[30] * vec3(0.00346457, 0.00125930, 0.00000000) +
            R[31] * vec3(0.00149761, 0.00054165, 0.00000000) +
            R[32] * vec3(0.00076970, 0.00027795, 0.00000000) +
            R[33] * vec3(0.00040737, 0.00014711, 0.00000000) +
            R[34] * vec3(0.00016901, 0.00006103, 0.00000000) +
            R[35] * vec3(0.00009522, 0.00003439, 0.00000000) +
            R[36] * vec3(0.00004903, 0.00001771, 0.00000000) +
            R[37] * vec3(0.00002000, 0.00000722, 0.00000000);
}

vec3 mixSpectral(vec3 A, vec3 B, float t) {
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

    // spectral_linear_to_concentration
    float t1 = l1 * pow(1.0 - t, 2.0);
    float t2 = l2 * pow(t, 2.0);
    t = t2 / (t1 + t2);

    float R[MIXSPECTRAL_SIZE];

    for (int i = 0; i < MIXSPECTRAL_SIZE; i++) {
        float KS = (1.0 - t) * (pow(1.0 - R1[i], 2.0) / (2.0 * R1[i])) + t * (pow(1.0 - R2[i], 2.0) / (2.0 * R2[i]));
        float KM = 1.0 + KS - sqrt(pow(KS, 2.0) + 2.0 * KS);
        R[i] = KM;
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