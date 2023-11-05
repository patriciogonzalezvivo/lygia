/*
contributors:  Inigo Quiles
description: generate the SDF of a cylinder
use: 
    - <float> cylinderSDF( in <vec3> pos, in <vec2|float> h [, <float> r] ) 
    - <float> cylinderSDF( <vec3> p, <vec3> a, <vec3> b, <float> r) 
*/

#ifndef FNC_CYLINDERSDF
#define FNC_CYLINDERSDF

// vertical
float cylinderSDF( vec3 p, vec2 h ) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float cylinderSDF( vec3 p, float h ) {
    return cylinderSDF( p, vec2(h) );
}

float cylinderSDF( vec3 p, float h, float r ) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

// arbitrary orientation
float cylinderSDF(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float baba = dot(ba,ba);
    float paba = dot(pa,ba);

    float x = length(pa*baba-ba*paba) - r*baba;
    float y = abs(paba-baba*0.5)-baba*0.5;
    float x2 = x*x;
    float y2 = y*y*baba;
    float d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
    return sign(d)*sqrt(abs(d))/baba;
}

#endif