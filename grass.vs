#version 150

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

in vec3 position;
out vec2 texCoord;

void main()
{
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4( position, 1.0 );
	texCoord = position.xz;
}

