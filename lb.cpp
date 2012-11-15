#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
//#include <cutil_inline.h>
//#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>

const unsigned int window_width = 1024;
const unsigned int window_height = 768;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;


int mouse_old_x, mouse_old_y, mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0, translate_z = -3.0;

bool fscreen = false;
StopWatchInterface *timer;
int add = 1;
float *d_fin, *d_fout;

extern "C" void launch_kernel(float4* pos, unsigned int mesh_width, unsigned int mesh_height, int add, float *d_fin, float *d_fout);

bool init(int argc, char** argv);
void cleanup();

bool initGL(int *argc, char** argv);
void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res);

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

void initArrays();
void runCuda(struct cudaGraphicsResource **vbo_resource);
void checkResultCuda(int argc, char** argv, const GLuint& vbo);

int main(int argc, char** argv)
{
    init(argc, argv);
    cudaThreadExit();
	cudaDeviceReset();
    exit(0);
}
						 /*
void computeFPS()
{
	char fps[256];
    float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
    sprintf(fps, "Lattice Boltzmann - Cuda GL Interop (VBO): %3.1f fps", ifps);  
    glutSetWindowTitle(fps);
 }					   */

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

	return true;
}

bool init(int argc, char** argv)
{
    sdkCreateTimer( &timer );
    
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
		
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
	cudaMalloc((void**)&d_fin, mesh_width * mesh_height * 9 * sizeof(float));
	cudaMalloc((void**)&d_fout, mesh_width * mesh_height * 9 * sizeof(float));
    
	atexit(cleanup);
	glutMainLoop();

    cudaThreadExit();

	return true;
}

void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    float4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes; 
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource));

	launch_kernel(dptr, mesh_width, mesh_height, add, d_fin, d_fout);
	add = 0;
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	
	unsigned int size = mesh_width * mesh_height * 8 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
}

void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res)
{
	cudaGraphicsUnregisterResource(vbo_res);
	
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	
	*vbo = 0;
}

void display()
{
    sdkStartTimer(&timer);

    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
	
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
	glColorPointer(4, GL_FLOAT, 0, (GLvoid *) (mesh_width * mesh_height * sizeof(float)*4));

    glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
    //glColor3f(0.0, 0.5, 1.0);
    
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();

    sdkStopTimer(&timer);  
    //computeFPS();
}

void cleanup()
{
    sdkDeleteTimer( &timer);
    deleteVBO(&vbo, cuda_vbo_resource);
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
