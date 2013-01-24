#version 150

in float height;
out vec4 outColor;
in vec3 texCoord;
uniform sampler2D tex;

void main()
{
    //outColor = vec4( 1.0, height+1., 0.0, 1.0 );
    //outColor = vec4(texCoord.x, texCoord.y, texCoord.z, 1.0);
    //outColor = texture(tex, texCoord.xy);
    if (texCoord.x < -0.9999f || texCoord.x > 0.9999f)
        outColor = texture(tex, texCoord.yz);
    else if (texCoord.y < -0.9999f || texCoord.y > 0.9999f)
        outColor = texture(tex, texCoord.xz);
    else 
        outColor = texture(tex, texCoord.xy);

}
