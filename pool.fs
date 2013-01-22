#version 150

in float height;
out vec4 outColor;

void main()
{
    outColor = vec4( 1.0, height+1., 0.0, 1.0 );
}
