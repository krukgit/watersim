#version 150

in float height;
out vec4 outColor;
in vec3 texCoord;
uniform sampler2D tex;

void main()
{
    //outColor = vec4( 1.0, height+1., 0.0, 1.0 );
    outColor = texture(tex, texCoord.xy);
}
