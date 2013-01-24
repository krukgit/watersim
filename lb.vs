#version 150

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

in vec4 vertex;
in vec4 color;
//in vec3 normals;

out vec4 mycolor;
out vec3 worldSpaceNormal;

void main()
{
	float heightScale = 0.5;
	vec2 size = vec2(256.0f, 256.0f);
	vec3 normal = normalize(cross( vec3(0.0, color.y*heightScale, 2.0 / size.x), vec3(2.0 / size.y, color.x*heightScale, 0.0)));
    
	gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(vertex.rgb, 1.0);  

	//vec4 pos         = vec4(gl_Vertex.x, height * heightScale, gl_Vertex.z, 1.0);
   // gl_Position      = gl_ModelViewProjectionMatrix * pos;
    
	worldSpaceNormal = normal.xyz;
    //vec3 eyeSpacePos  = (viewMatrix*modelMatrix * vertex).xyz;
    //vec3 eyeSpaceNormal   = (gl_NormalMatrix * normal).xyz;

	//gl_Position.y -= 1.0;
	//mat4 m(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	//gl_Position = m*vec4(vertex.rgb,1.0);
    //gl_Position = vec4(vertex.rgb,1.0);
    mycolor = vec4(0.47, 0.37, 0.28, 0.5);
	//mycolor = vec4(0.0, 0.0, 1.0, 1.0); 
}