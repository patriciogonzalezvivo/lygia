#include "../luminance.hlsl"

/*
contributors: nan
description: |
    Converts the input HDR RGB color into one of 16 debug colors that represent
    the pixel's exposure. When the output is cyan, the input color represents
    middle gray (18% exposure). Every exposure stop above or below middle gray
    causes a color shift.
 
    The relationship between exposures and colors is:
 
    -5EV  - black
    -4EV  - darkest blue
    -3EV  - darker blue
    -2EV  - dark blue
    -1EV  - blue
     OEV  - cyan
    +1EV  - dark green
    +2EV  - green
    +3EV  - yellow
    +4EV  - yellow-orange
    +5EV  - orange
    +6EV  - bright red
    +7EV  - red
    +8EV  - magenta
    +9EV  - purple
    +10EV - white

use: <float3|float4> tonemapDebug(<float3|float4> x)
*/

#ifndef FNC_TONEMAPDEBUG
#define FNC_TONEMAPDEBUG

#if !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
float3 tonemapDebug(const float3 x) {
    
    // 16 debug colors + 1 duplicated at the end for easy indexing
    float3 debugColors[17];
    debugColors[0] = float3(0.0, 0.0, 0.0);         // black
    debugColors[1] = float3(0.0, 0.0, 0.1647);      // darkest blue
    debugColors[2] = float3(0.0, 0.0, 0.3647);      // darker blue
    debugColors[3] = float3(0.0, 0.0, 0.6647);      // dark blue
    debugColors[4] = float3(0.0, 0.0, 0.9647);      // blue
    debugColors[5] = float3(0.0, 0.9255, 0.9255);   // cyan
    debugColors[6] = float3(0.0, 0.5647, 0.0);      // dark green
    debugColors[7] = float3(0.0, 0.7843, 0.0);      // green
    debugColors[8] = float3(1.0, 1.0, 0.0);         // yellow
    debugColors[9] = float3(0.90588, 0.75294, 0.0); // yellow-orange
    debugColors[10] = float3(1.0, 0.5647, 0.0);      // orange
    debugColors[11] = float3(1.0, 0.0, 0.0);         // bright red
    debugColors[12] = float3(0.8392, 0.0, 0.0);      // red
    debugColors[13] = float3(1.0, 0.0, 1.0);         // magenta
    debugColors[14] = float3(0.6, 0.3333, 0.7882);   // purple
    debugColors[15] = float3(1.0, 1.0, 1.0);         // white
    debugColors[16] = float3(1.0, 1.0, 1.0);         // white

    // The 5th color in the array (cyan) represents middle gray (18%)
    // Every stop above or below middle gray causes a color shift
    float v = log2(luminance(x) / 0.18);
    v = clamp(v + 5.0, 0.0, 15.0);
    int index = int(v);
    return lerp(debugColors[index], debugColors[index + 1], v - float(index));
}
float4 tonemapDebug(const float4 x) { return float4(tonemapDebug(x.rgb), x.a); }
#endif

#endif