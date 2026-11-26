#version 450
// The triangle, turned and moved on the CPU's numbers: xost's sin and cos are only good from -pi to pi, so the
// rotation comes in as its cosine and sine (xsdk docs/PITFALLS.md, "xost's GLSL subset").
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 colour;

layout(push_constant) uniform Transform {
	vec4 rotation; // cos, sin, scale, unused
	vec4 offset;   // x, y, unused, unused
} transform;

layout(location = 0) out vec3 v_colour;

void main()
{
	vec2 p = vec2(position.x * transform.rotation.x - position.y * transform.rotation.y,
		position.x * transform.rotation.y + position.y * transform.rotation.x);
	gl_Position = vec4(p * transform.rotation.z + transform.offset.xy, 0.0, 1.0);
	v_colour = colour;
}
