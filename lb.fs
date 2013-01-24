#version 150
in vec4 mycolor;
in vec3 worldSpaceNormal;

out vec4 fragmentColor;
void main()
{
    vec3 lightDir = vec3(0., 1., 1.);
    float diffuse   = max(0.0, dot(worldSpaceNormal, lightDir));
    fragmentColor = mycolor*diffuse;
    fragmentColor = vec4(fragmentColor.rgb, fragmentColor.a*1.5f);
    
}
