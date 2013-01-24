#version 150

in vec2 texCoord;
out vec4 outColor;
uniform sampler2D tex;

void main()
{
    outColor = texture(tex, texCoord);
    //outColor = vec4( 0.0, 1.0, 0.0, 1.0 );
}
