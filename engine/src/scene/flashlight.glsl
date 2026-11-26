// Analytic projection profile shared by walls, props and skinned meshes.
// Keep the whole cone calculation highp: mediump cosines near 1 quantize a
// narrow beam into rings on GLES, even when the world positions are highp.
// One return: the Xbox 360's shader compiler inlines only single-exit functions.
highp float flashlightCone(highp float cosine, highp float innerCosine, highp float outerCosine) {
    highp float cone = 0.0;
    if (cosine > 0.0) {
        // Squared radius on a plane one unit in front of the lamp. A radial
        // profile needs no projection basis, texture fetch or resolution limit.
        highp float radius2 = max(1.0 - cosine * cosine, 0.0) / (cosine * cosine);
        highp float inner2 = max(1.0 / (innerCosine * innerCosine) - 1.0, 0.000001);
        highp float outer2 = max(1.0 / (outerCosine * outerCosine) - 1.0, inner2 + 0.000001);
        highp float hotspot = 1.0 - smoothstep(0.0, inner2, radius2);
        highp float spill = 1.0 - smoothstep(inner2, outer2, radius2);
        cone = 0.65 * hotspot + 0.35 * spill;
    }
    return cone;
}

highp float flashlightDistance(highp float distance, highp float range) {
    // Fade the last quarter of the reach instead of clipping a bright disc
    // against walls at the range boundary.
    highp float fade = 1.0 - smoothstep(range * 0.75, range, distance);
    return fade / (1.0 + 0.1 * distance + 0.01 * distance * distance);
}
