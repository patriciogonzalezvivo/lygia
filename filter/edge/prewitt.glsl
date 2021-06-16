/*
author: Brad Larson
description: Adapted version of Prewitt edge detection from https://github.com/BradLarson/GPUImage2
use: edgePrewitt(<sampler2D> texture, <vec2> st, <vec2> scale)
options:
    EDGEPREWITT_TYPE: Return type, defaults to float
    EDGEPREWITT_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,POS_UV).r
license: |
    Copyright (c) 2015, Brad Larson.
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met

    Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef EDGEPREWITT_TYPE
#define EDGEPREWITT_TYPE float
#endif

#ifndef EDGEPREWITT_SAMPLER_FNC
#define EDGEPREWITT_SAMPLER_FNC(POS_UV) texture2D(tex,POS_UV).r
#endif

#ifndef FNC_EDGEPREWITT
#define FNC_EDGEPREWITT
EDGEPREWITT_TYPE edgePrewitt(in sampler2D tex, in vec2 st, in vec2 offset) {
    // get samples around pixel
    EDGEPREWITT_TYPE tleft = EDGEPREWITT_SAMPLER_FNC(st + vec2(-offset.x, offset.y));
    EDGEPREWITT_TYPE left = EDGEPREWITT_SAMPLER_FNC(st + vec2(-offset.x, 0.));
    EDGEPREWITT_TYPE bleft = EDGEPREWITT_SAMPLER_FNC(st + vec2(-offset.x, -offset.y));
    EDGEPREWITT_TYPE top = EDGEPREWITT_SAMPLER_FNC(st + vec2(0., offset.y));
    EDGEPREWITT_TYPE bottom = EDGEPREWITT_SAMPLER_FNC(st + vec2(0., -offset.y));
    EDGEPREWITT_TYPE tright = EDGEPREWITT_SAMPLER_FNC(st + offset);
    EDGEPREWITT_TYPE right = EDGEPREWITT_SAMPLER_FNC(st + vec2(offset.x, 0.));
    EDGEPREWITT_TYPE bright = EDGEPREWITT_SAMPLER_FNC(st + vec2(offset.x, -offset.y));
    EDGEPREWITT_TYPE x = -tleft - top - tright + bleft + bottom + bright;
    EDGEPREWITT_TYPE y = -bleft - left - tleft + bright + right + tright;
    return sqrt((x * x) + (y * y));
}
#endif
