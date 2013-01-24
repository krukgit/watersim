#version 150

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

in vec3 position;
out float height;
out vec3 texCoord;

void main()
{
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4( position, 1.0 );
	height = position.y;
	texCoord = position.xyz;
	/*if (texCoord.x == 0.0f || texCoord.x == 1.0f)
		texCoord = position.yz*0.5f+0.5f;
	else if (texCoord.y == 0.0f || texCoord.y == 1.0f)
		texCoord = position.xz*0.5f+0.5f;
	else
		texCoord = position.xy*0.5f+0.5f;
		*/
	//;texCoord = position;
}

