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

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>

const unsigned int window_width = 1024;
const unsigned int window_height = 768;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

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
void runCuda(struct cudaGraphicsResource **vbo_resource);

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

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
	
    glBindBuffer(GL_ARRAY_BUFFER, height_map);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, colors);
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(4, GL_FLOAT, 0, 0);
    
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
