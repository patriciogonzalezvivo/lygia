/*
description: Returns the cartesian to polar coordinates
use: spherical(<vec3> st)
*/

#ifndef FNC_CART2POLARXYZ
#define FNC_CART2POLARXYZ
vec3 cart2polarXYZ( in vec3 st ) 
{
   float r = length(st);
   float theta = atan(length(st.xy), st.z);
   float phi = atan(st.y, st.x);
   vec3 w = vec3(r, theta, phi);
   return w;
}
#endif