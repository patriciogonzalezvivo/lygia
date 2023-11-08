#include "../math/const.glsl"

/*
contributors: [Mario Carrillo, Daniel Ilett]
description:
    Converts carteesian coordinates into polar coordinates 
    emulating a kaleidoscope visual effect.
    Based on Daniel Ilett's tutorial on reflecting polar coordinates: 
        https://danielilett.com/2020-02-19-tut3-8-crazy-kaleidoscopes/
*/

// Default use when just the coordinates are given, the segment count is set to 8
float2 kaleidoscope(float2 coord) {
    float segmentCount = 8.0;
    
    float2 uv = coord - 0.5;
    float radius = length(uv);
    float angle = atan2(uv.y, uv.x);
    
    float segmentAngle = TWO_PI / segmentCount;
    angle -= segmentAngle * floor(angle / segmentAngle);
    angle = min(angle, segmentAngle - angle);    
    
    float2 kuv = float2(cos(angle), sin(angle)) * radius + 0.5;  
    kuv = max(min(st, 2.0 - st), -st);  

    return kuv;
}

// Use when when you want to specify the segment count
float2 kaleidoscope(float2 coord, float segmentCount) {    
    float2 uv = coord - 0.5;
    float radius = length(uv);
    float angle = atan2(uv.y, uv.x);
    
    float segmentAngle = TWO_PI / segmentCount;
    angle -= segmentAngle * floor(angle / segmentAngle);
    angle = min(angle, segmentAngle - angle);    
    
    float2 kuv = float2(cos(angle), sin(angle)) * radius + 0.5;  
    kuv = max(min(st, 2.0 - st), -st);  

    return kuv;
}

// Use when when you want to specify the segment count and the phase so it helps with animation
float2 kaleidoscope(float2 coord, float segmentCount, float phase) {    
    float2 uv = coord - 0.5;
    float radius = length(uv);
    float angle = atan2(uv.y, uv.x);
    
    float segmentAngle = TWO_PI / segmentCount;
    angle -= segmentAngle * floor(angle / segmentAngle);
    angle = min(angle, segmentAngle - angle);    
    
    float2 kuv = float2(cos(angle + phase), sin(angle + phase)) * radius + 0.5;  
    kuv = max(min(st, 2.0 - st), -st);  

    return kuv;
}