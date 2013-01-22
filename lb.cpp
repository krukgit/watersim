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

int mouse_old_x, mouse_old_y, mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0, translate_x = 0.0, translate_y = -0.2, translate_z = -3.0;

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
GLuint grass;
GLuint pool;

int numGrassVertices;
int numPoolVertices;

void createVBO(GLuint* vbo);
void createIndexMap(GLuint* vbo);
void deleteVBO(GLuint* vbo);

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

void initArrays();
void initialize();
GLhandleARB shaderProgramBuild(const GLchar *vertex, const GLchar *fragment, int attach);
//void runCuda(struct cudaGraphicsResource **vbo_resource);

GLhandleARB shaderProgram;
GLhandleARB shaderProgramGrass;
GLhandleARB shaderProgramPool;

GLchar *fragmentShaderSource, *vertexShaderSource;
GLchar *fragmentShaderSourceGrass, *vertexShaderSourceGrass;
GLchar *fragmentShaderSourcePool, *vertexShaderSourcePool;
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
	//FreeConsole();
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

bool checkErrors() {
switch ( glGetError() ) {
case GL_NO_ERROR: {
return true;
}
case GL_INVALID_ENUM: {
cout << "GL_INVALID_ENUM" << endl;
break;
}
case GL_INVALID_VALUE: {
cout << "GL_INVALID_VALUE" << endl;
break;
}
case GL_INVALID_OPERATION: {
cout << "GL_INVALID_OPERATION" << endl;
break;
}
case GL_INVALID_FRAMEBUFFER_OPERATION: {
cout << "GL_INVALID_FRAMEBUFFER_OPERATION" << endl;
break;
}
case GL_OUT_OF_MEMORY: {
cout << "GL_OUT_OF_MEMORY" << endl;
break;
}
}
return false;
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

	GLfloat _maxS = -1.0f;
	GLfloat _maxT = 1.0f;
	
	GLfloat _far = 100.0f;
	GLfloat _z = -0.2f;
	GLfloat _near = 1.0f;
	numGrassVertices = 13;
	GLfloat	 vertices[] = {  
		_far,	_z,		_far,		// 1
		-_far,	_z,		_far,		// 2
		_far,	_z,		_near,		// 3
		-_far,	_z,		_near,		// 4
		-_near,	_z,		_near,		// 5
		-_far,	_z,		-_far,		// 6
		-_near,	_z,		-_far,		// 7
		-_near,	_z,		-_near, 	// 8
		_far,	_z,		-_far,		// 9
		_far,	_z,		-_near, 	// 10
		_far,	_z,		_near,		// 11
		_near,	_z,		-_far, 	// 13
		
		_near,	_z,		_near		// 12
		
	};

	glGenBuffers( 1, &grass ); // Generate 1 buffer
	glBindBuffer( GL_ARRAY_BUFFER, grass );
	glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ), vertices, GL_DYNAMIC_DRAW );

	GLfloat bottom = -1.0;
	GLfloat one = 1.0;
	GLfloat top = _z;
	numPoolVertices = 12;
	GLfloat	 poolVertices[] = {  
		-one, bottom, -one,
		one, bottom, -one,
		-one, bottom, one,
		one, bottom, one,
		one, top, one,
		one, bottom, -one,
		one, top, -one,
		-one, bottom, -one,
		-one, top, -one,
		-one, bottom, one,
		-one, top, one,
		one, top, one
	};

	glGenBuffers( 1, &pool); // Generate 1 buffer
	glBindBuffer( GL_ARRAY_BUFFER, pool );
	glBufferData( GL_ARRAY_BUFFER, sizeof( poolVertices ), poolVertices, GL_DYNAMIC_DRAW );

	

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
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendColor(0.0, 0.0, 1.0, 0.5);

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

void drawWater() {
	glUseProgram(shaderProgram);
	//viewMatrix = glm::translate(viewMatrix, glm::vec3(0.0f, 0.0f, -1.0f)); // Create our view matrix which will translate us back 5 units  	
	int projectionMatrixLocation = glGetUniformLocation(shaderProgram, "projectionMatrix");
	int viewMatrixLocation = glGetUniformLocation(shaderProgram, "viewMatrix");
	int modelMatrixLocation = glGetUniformLocation(shaderProgram, "modelMatrix");
	glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]); // Send our projection matrix to the shader  
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]); // Send our view matrix to the shader  
	glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]); // Send our model matrix to the shader  
	
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBindBuffer(GL_ARRAY_BUFFER, height_map);
	glEnableVertexAttribArray(height_map);
	glVertexAttribPointer(height_map, 4, GL_FLOAT, GL_FALSE, sizeof(float)*4, 0);

	glBindBuffer(GL_ARRAY_BUFFER, colors);
	glEnableVertexAttribArray(colors);
	glVertexAttribPointer(colors, 4, GL_FLOAT, GL_FALSE, sizeof(float)*4, 0);

    glBindVertexArray(height_map);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawElements(GL_TRIANGLE_STRIP, mesh_width*2*(mesh_height-1)+mesh_height-2, GL_UNSIGNED_INT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void drawPool() {
	glUseProgram(shaderProgramPool);
	int projectionMatrixLocation = glGetUniformLocation(shaderProgramPool, "projectionMatrix");
	int viewMatrixLocation = glGetUniformLocation(shaderProgramPool, "viewMatrix");
	int modelMatrixLocation = glGetUniformLocation(shaderProgramPool, "modelMatrix");
	glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]); 
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]); 
	glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]); 

	//checkErrors();

	glBindVertexArray(pool);
	glBindBuffer(GL_ARRAY_BUFFER, pool);
	glEnableVertexAttribArray( pool );
	glVertexAttribPointer(pool, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
	glDrawArrays( GL_TRIANGLE_STRIP, 0, numPoolVertices );

}

void drawGrass() {
	glUseProgram(shaderProgramGrass);
	int projectionMatrixLocation = glGetUniformLocation(shaderProgramGrass, "projectionMatrix");
	int viewMatrixLocation = glGetUniformLocation(shaderProgramGrass, "viewMatrix");
	int modelMatrixLocation = glGetUniformLocation(shaderProgramGrass, "modelMatrix");
	glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]); 
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]); 
	glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]); 

	//checkErrors();

	glBindVertexArray(grass);
	glBindBuffer(GL_ARRAY_BUFFER, grass);
	glEnableVertexAttribArray( grass );
	glVertexAttribPointer(grass, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
	glDrawArrays( GL_TRIANGLE_STRIP, 0, numGrassVertices );

}

void display()
{
    runCuda();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	    // added this static boolean do do one-time OpenGL related initialization.
    static bool doInitialize = true;

    if (doInitialize == true) {
        doInitialize = false;
    
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f) ;
		vertexShaderSource = read("lb.vs");
		fragmentShaderSource = read("lb.fs");
		shaderProgram = shaderProgramBuild(vertexShaderSource, fragmentShaderSource, 1);	

		vertexShaderSourceGrass = read("grass.vs");
		fragmentShaderSourceGrass = read("grass.fs");
		shaderProgramGrass = shaderProgramBuild(vertexShaderSourceGrass, fragmentShaderSourceGrass, 2);	

		vertexShaderSourcePool = read("pool.vs");
		fragmentShaderSourcePool = read("pool.fs");
		shaderProgramPool = shaderProgramBuild(vertexShaderSourcePool, fragmentShaderSourcePool, 3);	
    }
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
	modelMatrix = glm::mat4(1.0f);
	viewMatrix = glm::mat4(1.0f);
	projectionMatrix = glm::mat4(1.0f);
	modelMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));  // Create our model matrix which will halve the size of our model  
	viewMatrix = glm::rotate(viewMatrix, rotate_x, glm::vec3(1.0f, 0.0f, 0.0f));
	viewMatrix = glm::rotate(viewMatrix, rotate_y, glm::vec3(0.0f, 1.0f, 0.0f));
	viewMatrix = glm::translate(viewMatrix, glm::vec3(0.0, -0.4, translate_z)); // Create our view matrix which will translate us back 5 units  
	projectionMatrix = glm::perspective(30.0f, (float)window_width / (float)window_height, 0.1f, 100.f);  // Create our perspective projection matrix  

	
	drawWater();
	drawPool();
	
	drawGrass();
	


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
		case(38): //UP
			translate_y += 1.;
			break;
		case(40): //DOWN
			translate_y -= 1.;
			break;
		case(37): //LEFT
			translate_x -= 1.;
			break;
		case(39): //RIGHT
			translate_x += 1.;
			break;
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

GLhandleARB shaderProgramBuild(const GLchar *vertex, const GLchar *fragment, int attach)
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
	
	if (attach == 1) {
		glBindAttribLocation(programHandle, colors, "color");
		glBindAttribLocation(programHandle, height_map, "vertex");
	} else if (attach == 2) {
		glBindAttribLocation(programHandle, grass, "position");
	} else if (attach == 3) {
			glBindAttribLocation(programHandle, pool, "position");
	}
    // Attempt to the link the shader objects into a program
    glLinkProgram(programHandle);
    // Check for any other errors and print them.
    glutReportErrors();

    // Ask GL if this source compiled.
    glGetProgramiv(programHandle, GL_LINK_STATUS, &status);

	//glBindAttribLocation(programHandle, height_map, "vertex");
    

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
    getchar();
    delete [] log;
    exit(1);
}
