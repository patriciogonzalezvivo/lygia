/*
original_author: Brad Larson
description: vibrance from https://github.com/BradLarson/GPUImage/blob/master/framework/Source/GPUImageVibranceFilter.m Vibrance is a smart-tool which cleverly increases the intensity of the more muted colors and leaves the already well-saturated colors alone. Prevents skin tones from becoming overly saturated and unnatural. 
use: <vec3|vec4> vibrance(<vec3|vec4> color, <float> v) 
license: |
    Copyright (c) 2012, Brad Larson, Ben Cochran, Hugues Lismonde, Keitaroh Kobayashi, Alaric Cole, Matthew Clark, Jacob Gundersen, Chris Williams.
    All rights reserved.
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    Neither the name of the GPUImage framework nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef FNC_VIBRANCE
#define FNC_VIBRANCE
vec3 vibrance(in vec3 color, in float v) {
    float average = (color.r + color.g + color.b) / 3.0;
    float mx = max(color.r, max(color.g, color.b));
    float amt = (mx - average) * (-v * 3.0);
    return mix(color.rgb, vec3(mx), amt);
}
vec4 vibrance(in vec4 color, in float v) { return vec4( vibrance(color.rgb, v), color.a); }
#endif