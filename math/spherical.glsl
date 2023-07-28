/*
description: Returns spherical coordinates
use: spherical(<float2> st)
*/

vec2 spherical( in vec2 st ) 
{
   float r = length(st);
   float theta = atan(st.y, st.x);
   vec2 w = vec2(r, theta);
   return w;
}