#include "../math/const.glsl"

/*
contributors: [Mario Carrillo, Daniel Ilett]
description:
    Transforms a carteesian coordinates into a kaleidoscope space.
    Based on Daniel Ilett's tutorial on reflecting polar coordinates: 
        https://danielilett.com/2020-02-19-tut3-8-crazy-kaleidoscopes/
*/

vec2 kaleidoscope(in vec2 coord) {
    float segmentCount = 8.0;
    
    vec2 uv = coord - 0.5;
    float radius = length(uv);
    float angle = atan(uv.y, uv.x);
    
    float segmentAngle = TWO_PI / segmentCount;
    angle -= segmentAngle * floor(angle / segmentAngle);
    angle = min(angle, segmentAngle - angle);    
    
    vec2 kuv = vec2(cos(angle), sin(angle)) * radius + 0.5;  
    kuv = max(min(st, 2.0 - st), -st);  

    return kuv;
}

vec2 kaleidoscope(in vec2 coord, in float segmentCount) {    
    vec2 uv = coord - 0.5;
    float radius = length(uv);
    float angle = atan(uv.y, uv.x);
    
    float segmentAngle = TWO_PI / segmentCount;
    angle -= segmentAngle * floor(angle / segmentAngle);
    angle = min(angle, segmentAngle - angle);    
    
    vec2 kuv = vec2(cos(angle), sin(angle)) * radius + 0.5;  
    kuv = max(min(st, 2.0 - st), -st);  

    return kuv;
}

vec2 kaleidoscope(in vec2 coord, in float segmentCount, in float phase) {    
    vec2 uv = coord - 0.5;
    float radius = length(uv);
    float angle = atan(uv.y, uv.x);
    
    float segmentAngle = TWO_PI / segmentCount;
    angle -= segmentAngle * floor(angle / segmentAngle);
    angle = min(angle, segmentAngle - angle);    
    
    vec2 kuv = vec2(cos(angle + phase), sin(angle + phase)) * radius + 0.5;  
    kuv = max(min(st, 2.0 - st), -st);  

    return kuv;
}