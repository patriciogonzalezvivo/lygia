/*
contributors:  Inigo Quiles
description: generate the SDF of a octahedron
use: <float> octahedronSDF(<vec3> p, <float> s)
options:
    OCTAHEDRON_EXACT_DISTANCE

*/

#ifndef FNC_OCTAHEDRONSDF
#define FNC_OCTAHEDRONSDF

float octahedronSDF(vec3 p, float s) {
    p = abs(p);
    float m = p.x + p.y + p.z - s;

#ifdef OCTAHEDRON_EXACT_DISTANCE
    // exact distance
    vec3 o = min(3.0*p - m, 0.0);
    o = max(6.0*p - m*2.0 - o*3.0 + (o.x+o.y+o.z), 0.0);
    return length(p - s*o/(o.x+o.y+o.z));

// #elif OCTAHEDRON_EXACT_DISTANCE == 2
//     // exact distance
//     vec3 q = vec3(0.0);
//          if( 3.0*p.x < m ) q = p.xyz;
//     else if( 3.0*p.y < m ) q = p.yzx;
//     else if( 3.0*p.z < m ) q = p.zxy;
//     else return m*0.57735027;
//     float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
//     return length(vec3(q.x,q.y-s+k,q.z-k)); 
    
#else
    // bound, not exact
    return m*0.57735027;
#endif
}

#endif