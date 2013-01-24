#include "common.h"
#include "GLProgram.h"

extern "C" void launch_kernel(float4* pos, float4* color, unsigned int mesh_width, unsigned int mesh_height, int add, float *d_fin, float *d_fout);
extern int add;
extern float *d_fin, *d_fout;
extern const unsigned int mesh_width, mesh_height;

class waterProgram : public GLProgram {
	GLuint height_map;
	GLuint colors;
	GLuint normals;
	GLuint indices;

public:
	waterProgram() {
		vertexShaderSource = read("lb.vs");
		fragmentShaderSource = read("lb.fs");
	}

	void bindVBOs(GLhandleARB programHandle) {
		glBindAttribLocation(programHandle, colors, "color");
		glBindAttribLocation(programHandle, height_map, "vertex");

	}

	void draw() {
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

	void prepare() {
		createVBO(&height_map);//, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
		createVBO(&colors);
		createIndexMap(&indices);

		cudaMalloc((void**)&d_fin, mesh_width * mesh_height * 9 * sizeof(float));
		cudaMalloc((void**)&d_fout, mesh_width * mesh_height * 9 * sizeof(float));
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

};
