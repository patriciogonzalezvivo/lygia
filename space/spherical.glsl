/*
author: Kathy McGuiness
description: Returns the spherical coordinates
use: sphericalSDF(<vec3> st)
*/

#ifndef FNC_SPHERICAL
#define FNC_SPHERICAL
vec3 spherical( in vec3 st) 
{
   float r = length(st);
   float theta = atan( length(st.xy), st.z);
   float phi = atan(st.y, st.x);
   vec3 w = vec3(r, theta, phi);
   return w;
}
#endif