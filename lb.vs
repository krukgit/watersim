#version 400

in vec4 inputPosition;
in vec4 inputColor;

out vec3 color;

void main(void)
{
	// Store the input color for the pixel shader to use.
	color = inputColor.rgb;
}
