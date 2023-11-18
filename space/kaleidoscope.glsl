#include "../math/const.glsl"

/*
contributors: [Mario Carrillo, Daniel Ilett]
description: |
    Converts carteesian coordinates into polar coordinates 
    emulating a kaleidoscope visual effect.
    Based on Daniel Ilett's tutorial on reflecting polar coordinates: 
    https://danielilett.com/2020-02-19-tut3-8-crazy-kaleidoscopes/
*/

// Use when when you want to specify the segment count and the phase for animation
vec2 kaleidoscope(in vec2 coord, in float segmentCount, in float phase) {    
    vec2 uv;

#ifdef CENTER_2D
    uv = coord - CENTER_2D;
#else
    uv = coord - 0.5;
#endif
    
    float radius = length(uv);
    float angle = atan(uv.y, uv.x);
    
    float segmentAngle = TWO_PI / segmentCount;
    angle -= segmentAngle * floor(angle / segmentAngle);
    angle = min(angle, segmentAngle - angle);    
    
    vec2 kuv = vec2(cos(angle + phase), sin(angle + phase)) * radius + 0.5;  
    kuv = max(min(kuv, 2.0 - kuv), -kuv);  

    return kuv;
}

// Default use when just the coordinates are given, the segment count is set to 8
vec2 kaleidoscope(in vec2 coord) {
    return kaleidoscope(coord, 8.0, 0.0);
}

// Use when when you want to specify the segment count
vec2 kaleidoscope(in vec2 coord, in float segmentCount) {    
    return kaleidoscope(coord, segmentCount, 0.0);
}
