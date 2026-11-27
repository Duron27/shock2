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

#ifdef SHOCK_XBOX360
// The Xbox 360's one dynamic light (slot 0), cut to what its GPU can afford
// across the screen: the same falloffs as calculateSpotlight without its
// highlight or shadows. The full light costs it about 16 ms a frame on the
// world alone (vulkan4xenon's vkreplay of earth.mis, 2026-11-26).
// falloffMode 0 is a player light (smooth cone, fade over its reach), 1 a
// mission lamp (authored linear cone, 1/distance from sourceRadius, lambert
// wrapped by wrap); a negative inner angle is a point light.
vec3 consoleLight(highp vec3 worldPos, vec3 normal, vec3 texColor, highp vec3 lightPos,
        vec4 colorIntensity, highp vec3 direction, highp float innerAngle, highp float outerAngle,
        highp float range, float falloffMode, float wrap, highp float sourceRadius) {
    highp vec3 lightVec = lightPos - worldPos;
    highp float d2 = max(dot(lightVec, lightVec), 1e-10);
    highp float inv = inversesqrt(d2);
    highp vec3 l = lightVec * inv;
    highp float distance = d2 * inv;
    bool lamp = falloffMode > 0.5;
    highp float cosOuter = cos(outerAngle);
    highp float cosInner = cos(innerAngle);
    // An empty slot's uniforms are zero, and both arms of a branch run on
    // this GPU: normalize(0) would be NaN, which no zero intensity cancels.
    highp float spot = dot(-l, direction * inversesqrt(max(dot(direction, direction), 1e-10)));
    highp float cone = lamp
        ? clamp((spot - cosOuter) / max(cosInner - cosOuter, 1e-5), 0.0, 1.0)
        : smoothstep(cosOuter, cosInner, spot);
    cone = innerAngle < 0.0 ? 1.0 : cone;
    highp float fade = lamp
        ? 1.0 / max(distance, sourceRadius)
        : (1.0 - smoothstep(range * 0.75, range, distance)) / (1.0 + 0.1 * distance + 0.01 * d2);
    fade = distance > range ? 0.0 : fade;
    float w = lamp ? wrap : 0.0;
    float lambert = max((dot(normal, l) + w) / (1.0 + w), 0.0);
    vec3 lit = texColor * colorIntensity.rgb * (colorIntensity.w * lambert * cone * fade);
    // A select, not a multiply: whatever an empty slot computed is dropped.
    return colorIntensity.w > 0.0 ? lit : vec3(0.0);
}
// The world's: its slot 0 is the player's light.
#define CONSOLE_FLASHLIGHT(worldPos, normal, texColor) consoleLight(worldPos, normal, texColor, \
    spotlightPos[0], spotlightColorIntensity[0], spotlightDirection[0], spotlightInnerAngle[0], \
    spotlightOuterAngle[0], spotlightRange[0], 0.0, 0.0, 1.0)
// Props' and creatures': their first light, a mission lamp as often as the
// player's, by a constant index (indexing by a loop counter is what hangs the
// GPU). Each more costs about 3 ms of this GPU a frame: the rest reach an
// object through its ambient instead (LightArray::console_folded_ambient).
#define CONSOLE_OBJECT_SLOT(i, worldPos, normal, texColor) consoleLight(worldPos, normal, texColor, \
    spotlightPos[i], spotlightColorIntensity[i], spotlightDirection[i], spotlightInnerAngle[i], \
    spotlightOuterAngle[i], spotlightRange[i], float(lightFalloffMode[i]), lambertWrap, SOURCE_RADIUS)
#define CONSOLE_OBJECT_LIGHT(worldPos, normal, texColor) \
    (CONSOLE_OBJECT_SLOT(0, worldPos, normal, texColor))
#endif
