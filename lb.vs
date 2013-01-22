#version 150

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

in vec4 vertex;
in vec4 color;

out vec4 mycolor;

void main()
{
    
	gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(vertex.rgb, 1.0);  
	//gl_Position.y -= 1.0;
	//mat4 m(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	//gl_Position = m*vec4(vertex.rgb,1.0);
    //gl_Position = vec4(vertex.rgb,1.0);
    mycolor = vec4(0.47, 0.37, 0.28, 0.5);
	//mycolor = vec4(0.0, 0.0, 1.0, 1.0); 
}