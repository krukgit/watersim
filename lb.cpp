#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>

const unsigned int window_width = 1024;
const unsigned int window_height = 768;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

glm::mat4 projectionMatrix; // Store the projection matrix  
glm::mat4 viewMatrix; // Store the view matrix  
glm::mat4 modelMatrix; // Store the model matrix  

//GLuint vbo;
//GLuint uiVBOIndices;

//struct cudaGraphicsResource *cuda_vbo_resource;

int mouse_old_x, mouse_old_y, mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0, translate_z = -3.0;

int mode=1;

bool fscreen = false;
int add = 1;
float *d_fin, *d_fout;

extern "C" void launch_kernel(float4* pos, float4* color, unsigned int mesh_width, unsigned int mesh_height, int add, float *d_fin, float *d_fout);

bool init(int argc, char** argv);
void cleanup();

bool initGL(int *argc, char** argv);

GLuint height_map;
GLuint colors;
GLuint normals;
GLuint indices;

void createVBO(GLuint* vbo);
void createIndexMap(GLuint* vbo);
void deleteVBO(GLuint* vbo);

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

void initArrays();
void initialize();
GLhandleARB shaderProgramBuild(const GLchar *vertex, const GLchar *fragment);
//void runCuda(struct cudaGraphicsResource **vbo_resource);

struct State {
    GLhandleARB shaderProgram;
};
State state;
/*
GLchar vertexShaderSource[] = "\n\
#version 150\n\
in vec4 vertex;\n\
in vec4 color;\n\
out vec4 mycolor;\n\
void\n\
main()\n\
{\n\
vec4 x(1.0, 0.0, 0.0, 0.0);\n\
vec4 y(0.0, 1.0, 0.0, 0.0);\n\
vec4 z(0.0, 0.0, 1.0, 0.0);\n\
vec4 w(0.0, 0.0, 1.0, 0.0);\n\
mat4 m(x,y,z,w);\n\
    gl_Position = m*vec4(vertex.rgb,1.0);\n\
    mycolor = vec4(0.0, vertex.g*10., 1.0, 1.0); // not transforming the color, just passing it to the fragment shader\n\
}\n\
";
GLchar fragmentShaderSource[] = "\n\
#version 150\n\
in vec4 mycolor;\n\
out vec4 fragmentColor;\n\
void\n\
main()\n\
{\n\
    fragmentColor = mycolor;\n\
}\n\
";
*/
GLchar *fragmentShaderSource, *vertexShaderSource;
using namespace std;
GLchar* read(const char* file_name)
{
	ifstream infile (file_name,ifstream::binary);
	infile.seekg(0,ifstream::end);
	long size=infile.tellg();
	infile.seekg(0);

	// allocate memory for file content
	char* buffer = new char [size];

	// read content of infile
	infile.read (buffer,size);
	infile.close();
	for (int i=size; i>=0; i--)
		if (buffer[i] == '}')
		{
			buffer[i+1] = 0;
			break;
		}
	return buffer;
}


int main(int argc, char** argv)
{
    init(argc, argv);
    cudaThreadExit();
	cudaDeviceReset();
    exit(0);
}		   

bool initShaders() 
{
	
	const char * my_fragment_shader_source = "lb.fs";
	const char * my_vertex_shader_source = "lb.ps";

	GLenum my_program;
	GLenum my_vertex_shader;
	GLenum my_fragment_shader;
	GLenum my_pixel_shader;
 
	// Create Shader And Program Objects
	my_program = glCreateProgramObjectARB();
	my_vertex_shader = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
	my_fragment_shader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
 
	// Load Shader Sources
	glShaderSourceARB(my_vertex_shader, 1, &my_vertex_shader_source, NULL);
	glShaderSourceARB(my_fragment_shader, 1, &my_fragment_shader_source, NULL);
 
	// Compile The Shaders
	glCompileShaderARB(my_vertex_shader);
	glCompileShaderARB(my_fragment_shader);
 
	// Attach The Shader Objects To The Program Object
	glAttachObjectARB(my_program, my_vertex_shader);
	glAttachObjectARB(my_program, my_fragment_shader);
 
	// Link The Program Object
	glLinkProgramARB(my_program);
 
	// Use The Program Object Instead Of Fixed Function OpenGL
	glUseProgramObjectARB(my_program);

	return true;
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Lattice Boltzmann - Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);

#ifdef _WIN32
	FreeConsole();
#endif
//	glutFullScreen();
	
	glewInit();

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, window_width, window_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

	//initShaders();
	return true;
}

bool init(int argc, char** argv)
{
    //sdkCreateTimer( &timer );
    
	if (false == initGL(&argc, argv))
    {
        cudaDeviceReset();
        return false;
    }

    findCudaGLDevice(argc, (const char **)argv);
		
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
		
	createVBO(&height_map);//, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
	createVBO(&colors);
	createIndexMap(&indices);

	

	/*glEnable(GL_LIGHTING);
	// Set Diffuse color component
	GLfloat LightColor [] = { 0.0f,0.0f,1.0f,0.5f };  
	// white
	glLightfv(GL_LIGHT0, GL_DIFFUSE, LightColor);
	// Set Position
	GLfloat LightPos[] = { 2.0f, 2.0f, 2.0f, 0.5f};
	glLightfv(GL_LIGHT0, GL_POSITION, LightPos);
	// Turn it on
	glEnable(GL_LIGHT0);
	
	glShadeModel(GL_SMOOTH);*/
//	glEnable(GL_BLEND);
//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


	cudaMalloc((void**)&d_fin, mesh_width * mesh_height * 9 * sizeof(float));
	cudaMalloc((void**)&d_fout, mesh_width * mesh_height * 9 * sizeof(float));
    
	atexit(cleanup);
	glutMainLoop();

    cudaThreadExit();

	return true;
}

void runCuda()
{
    float4 *hmptr, *cptr;
	cudaGLMapBufferObject((void**)&hmptr, height_map);
	cudaGLMapBufferObject((void**)&cptr, colors);
	
	launch_kernel(hmptr, cptr, mesh_width, mesh_height, add, d_fin, d_fout);
	add = 0;
	cudaGLUnmapBufferObject(colors);
	cudaGLUnmapBufferObject(height_map);
}

void createVBO(GLuint* vbo) {
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	cudaGLRegisterBufferObject(*vbo);
}

void deleteVBO(GLuint* vbo) {
	cudaGLUnregisterBufferObject(*vbo);	
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	
	*vbo = 0;
}

void createIndexMap(GLuint *vbo) {
	// indices 
	int numIndices = mesh_width*2*(mesh_height-1)+mesh_height-2;
	glGenBuffersARB(1, vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices*sizeof(GLuint), 0, GL_STATIC_DRAW);
	GLuint *iIndices = (GLuint*) glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
	
	int i=0;
	int restartIndex = mesh_width*mesh_height;
	for(int y=0; y<mesh_height-1; y++)
	{
		int currentIndex = y*mesh_width;
		for (int x=0; x<mesh_width; x++)
		{
			iIndices[i++] = currentIndex;
			iIndices[i++] = currentIndex+mesh_width;
			currentIndex++;														
		}
		if (y < mesh_height-2)
			iIndices[i++] = restartIndex;
	}
	
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(mesh_width*mesh_height);
	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void display()
{
    runCuda();

	    // added this static boolean do do one-time OpenGL related initialization.
    static bool doInitialize = true;

    if (doInitialize == true) {
        doInitialize = false;
    
		glClearColor(0.4f, 0.0f, 0.0f, 1.0f) ;
		vertexShaderSource = read("lb.vs");
		fragmentShaderSource = read("lb.fs");
		state.shaderProgram = shaderProgramBuild(vertexShaderSource, fragmentShaderSource);	
    }
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
	glUseProgram(state.shaderProgram);
	
	
	projectionMatrix = glm::perspective(60.0f, (float)window_width / (float)window_height, 0.1f, 100.f);  // Create our perspective projection matrix  
	viewMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, translate_z)); // Create our view matrix which will translate us back 5 units  
	viewMatrix = glm::rotate(viewMatrix, rotate_x, glm::vec3(1.0f, 0.0f, 0.0f));
	viewMatrix = glm::rotate(viewMatrix, rotate_y, glm::vec3(0.0f, 1.0f, 0.0f));
	modelMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));  // Create our model matrix which will halve the size of our model  
	
		int projectionMatrixLocation = glGetUniformLocation(state.shaderProgram, "projectionMatrix");
		int viewMatrixLocation = glGetUniformLocation(state.shaderProgram, "viewMatrix");
		int modelMatrixLocation = glGetUniformLocation(state.shaderProgram, "modelMatrix");
		glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]); // Send our projection matrix to the shader  
		glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]); // Send our view matrix to the shader  
		glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]); // Send our model matrix to the shader  
	
    
	//float scale = 0.5;
	//int scaleLocation = glGetUniformLocation(state.shaderProgram, "scale");
	//glUniform1f(scaleLocation, scale);
	
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
	

    glBindBuffer(GL_ARRAY_BUFFER, height_map);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, 0);
    //glVertexAttribPointer(height_map, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, colors);
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(4, GL_FLOAT, 0, 0);
	//glVertexAttribPointer(colors, 4, GL_FLOAT, GL_FALSE, 0, 0);

    //glEnableVertexAttribArray(height_map);
    //glEnableVertexAttribArray(colors);

    
	//glBindBuffer(GL_COLOR_ARRAY, colors);
	//glColorPointer(4, GL_FLOAT, 0, (GLvoid*));
	//glColorPointer(4, GL_FLOAT, 0, (GLvoid *)0);// (mesh_width * mesh_height * sizeof(float)*4));

    //glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
    glColor3f(0.0, 0.5, 1.0);
    glBindVertexArray(height_map);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawElements(GL_TRIANGLE_STRIP, mesh_width*2*(mesh_height-1)+mesh_height-2, GL_UNSIGNED_INT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	//glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();

	glUseProgram(0);

    glFlush();

	glutReportErrors();
	//return ;


    //computeFPS();
}

void cleanup()
{
    deleteVBO(&height_map);
	deleteVBO(&indices);

}


void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	if (key >= 48 && key <=57)
		add |= int(pow(2.0, key-48));
    switch(key) {
		case(27) : // ESC
			exit(0);
			break;
		case(32) :
			fscreen = !fscreen;
			if (fscreen)
				glutFullScreen();
			else
			    glutReshapeWindow(window_width, window_height);
	}
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}


GLhandleARB
shaderCompile(const GLchar *text, GLenum type)
{
    GLhandleARB shaderHandle;
    GLint       status;
    GLint       loglen;
    GLint       maxloglen;
    GLchar     *log = 0;
   
    // Create a shader handle.
    // This gives us a way of setting up a new shader.
   shaderHandle = glCreateShader(type);

   // Check for error and exit if we find one.
    if (shaderHandle == 0) {
        fprintf(stderr,"glCreateShader() failed.\n");
        fprintf(stderr,"Possibly outside an OpenGL context. Possible out of resources.\n");
        exit(1);
    }

    // Specify the source code for the shader. 
    // Shader text can be specified as either a single string of GLchars
    // or an array of character strings. Below I specify the shader source
    // as a single, NULL terminated array of strings
    glShaderSource(shaderHandle, 1, &text, NULL);
    // Check for errors and print them
    glutReportErrors();

    // Ask GL to compile the shader into object code
    glCompileShader(shaderHandle);
    // Check for errors and print them
    glutReportErrors();

    // Ask GL if this source compiled.
    glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &status);

    // If status is not 0, then the source code successfully 
    // complied into object code and we return the handle we originally 
    // created.
    if (status != 0) {
        return shaderHandle;
    }

    // otherwise we have a compilation error. 
    // So we get the error log, print the error log, and print out
    // the text of the offending shader. Then we exit. 

    // Get the error log length
    // log length includes the c string null terminator
    glGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH, &maxloglen);
    log = new GLchar[maxloglen];

    // get the error log and store it in 'log'
    glGetShaderInfoLog(shaderHandle, maxloglen, &loglen, log);

    fprintf(stderr,"compile failed.\n");
    fprintf(stderr,"shader text:\n");
    fprintf(stderr,"------------\n");
    fprintf(stderr,"%s\n",text);
    fprintf(stderr,"log text:\n");
    fprintf(stderr,"------------\n");
    fprintf(stderr,"%s\n",log);
    
    delete [] log;
    exit(1);
}

GLhandleARB shaderProgramBuild(const GLchar *vertex, const GLchar *fragment)
{
    GLhandleARB programHandle;
    GLint       status;
    GLint       loglen;
    GLint       maxloglen;
    GLchar     *log = 0;

    // Create a shader program. This again is an empty handle which we
    // will fill in with data.
    programHandle = glCreateProgram();
    
    // Exit if there is an error.
    if (programHandle < 1) {
        fprintf(stderr,"glCreateShader() failed.\n");
        exit(1);
    }

    // Check for any other errors and print them.
    glutReportErrors();

    // Attach the vertex shader object to the program.
    glAttachShader(programHandle, shaderCompile(vertex,GL_VERTEX_SHADER));
    // Attach the vertex shader object to the program.
    glAttachShader(programHandle, shaderCompile(fragment,GL_FRAGMENT_SHADER));
    // Check for any other errors and print them.
    glutReportErrors();

    // Attempt to the link the shader objects into a program
    glLinkProgram(programHandle);
    // Check for any other errors and print them.
    glutReportErrors();

    // Ask GL if this source compiled.
    glGetProgramiv(programHandle, GL_LINK_STATUS, &status);

    // If status is not 0, then the objects successfully 
    // linked into shader program and we return the handle we originally 
    // created.
    if (status != 0) {
        return programHandle;
    }

    // otherwise we have a link error. 
    // So we get the error log and print the error log.
    // Then we exit. 
    
    // Get the info log length
    // log length includes the c string null terminator
    glGetProgramiv(programHandle, GL_INFO_LOG_LENGTH, &maxloglen);
    log = new GLchar[maxloglen];

    // get the info log and store it in 'log'
    glGetProgramInfoLog(programHandle, maxloglen, &loglen, log);

    fprintf(stderr,"link failed.\n");
    fprintf(stderr,"log text:\n");
    fprintf(stderr,"------------\n");
    fprintf(stderr,"%s\n",log);
    
    delete [] log;
    exit(1);
}
