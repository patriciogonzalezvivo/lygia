/*
contributors:  Inigo Quiles
description: generate the SDF of a octogon prism
use: <float> octogonPrismSDF( in <vec3> p, in <float> r, <float> h )
*/

#ifndef FNC_OCTOGONPRISMSDF
#define FNC_OCTOGONPRISMSDF

float octogonPrismSDF( in vec3 p, in float r, float h ) {
   vec3 k = vec3( -0.9238795325,   // sqrt(2+sqrt(2))/2 
                  0.3826834323,   // sqrt(2-sqrt(2))/2
                  0.4142135623 ); // sqrt(2)-1 
   // reflections
   p = abs(p);
   p.xy -= 2.0*min(dot(vec2( k.x,k.y),p.xy),0.0)*vec2( k.x,k.y);
   p.xy -= 2.0*min(dot(vec2(-k.x,k.y),p.xy),0.0)*vec2(-k.x,k.y);
   // polygon side
   p.xy -= vec2(clamp(p.x, -k.z*r, k.z*r), r);
   vec2 d = vec2( length(p.xy)*sign(p.y), p.z-h );
   return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

#endif