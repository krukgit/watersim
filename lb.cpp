
#include "common.h"
#include "grass.cpp"
#include "pool.cpp"
#include "water.cpp"
#include "cubemap.h"

const unsigned int window_width = 1024;
const unsigned int window_height = 768;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

glm::mat4 projectionMatrix; // Store the projection matrix  
glm::mat4 viewMatrix; // Store the view matrix  
glm::mat4 modelMatrix; // Store the model matrix  

int mouse_old_x, mouse_old_y, mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0, translate_x = 0.0, translate_y = -0.2, translate_z = -3.0;
int add=1;
float *d_fin, *d_fout;
int mode=1;
bool fscreen = false;

bool init(int argc, char** argv);
bool initGL(int *argc, char** argv);

grassProgram grass;
poolProgram pool;
waterProgram water;
cubemapProgram cubemap;

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

void initArrays();
void initialize();
GLhandleARB shaderProgramBuild(const GLchar *vertex, const GLchar *fragment, int attach);
//void runCuda(struct cudaGraphicsResource **vbo_resource);
//GLuint vbo_cube_vertices;
GLhandleARB shaderProgram;
GLhandleARB shaderProgramCubemap;

GLchar *fragmentShaderSource, *vertexShaderSource;
GLchar *fragmentShaderSourceCubemap, *vertexShaderSourceCubemap;
using namespace std;


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
		
	grass.prepare();
	pool.prepare();	

	cubemap.prepare();
	//skybox = new SkyBox();
	
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
	//glBlendColor(0.0, 0.0, 1.0, 0.5);

	water.prepare();

	glutMainLoop();

    cudaThreadExit();

	return true;
}

void display()
{
	water.runCuda();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	
	    // added this static boolean do do one-time OpenGL related initialization.
    static bool doInitialize = true;

    if (doInitialize == true) {
        doInitialize = false;
    
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f) ;
	
		grass.compileShaders();
		pool.compileShaders();
		water.compileShaders();
		cubemap.compileShaders();

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
			checkErrors();

	pool.draw();
	grass.draw();

	water.draw();

	cubemap.draw();
	glutSwapBuffers();
    glutPostRedisplay();

	glUseProgram(0);

    glFlush();

	glutReportErrors();
	//return ;


    //computeFPS();
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