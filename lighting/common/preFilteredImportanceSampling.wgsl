fn prefilteredImportanceSampling(ipdf: f32, omegaP: f32, numSamples: i32) -> f32 {
    let K = 4.0;
    let omegaS = ipdf / float(numSamples);
    float mipLevel = log2(K * omegaS / omegaP) * 0.5; // log4
    return mipLevel;
}
