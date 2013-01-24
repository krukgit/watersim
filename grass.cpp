#include "common.h"
#include "GLProgram.h"

class grassProgram : public GLProgram {
		GLuint texture;

public:
	grassProgram() {
		GLfloat _far = 10.0f;
		GLfloat _z = -0.2f;
		GLfloat _near = 1.0f;
		numVertices = 13;
		vertices = new GLfloat[numVertices*3];
		GLfloat tmp[] = {  
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
		for (int i=0; i<numVertices*3; i++)
			this->vertices[i] = tmp[i];

		vertexShaderSource = read("grass.vs");
		fragmentShaderSource = read("grass.fs");
	}

	void prepare() {
		glGenBuffers( 1, &vbo ); // Generate 1 buffer
		glBindBuffer( GL_ARRAY_BUFFER, vbo );
		glBufferData( GL_ARRAY_BUFFER, numVertices*3*sizeof(GLfloat), vertices, GL_STATIC_DRAW );

		glGenTextures(1, &texture);
		glBindTexture( GL_TEXTURE_2D, texture );
		glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
		// when texture area is small, bilinear filter the closest mipmap
		glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                 GL_LINEAR_MIPMAP_NEAREST );
		// when texture area is large, bilinear filter the original
		glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		// the texture wraps over at the edges (repeat)
		glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
		glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );

		SDL_Surface *surf = IMG_Load("media/ground.jpg");
		gluBuild2DMipmaps( GL_TEXTURE_2D, surf->format->BytesPerPixel, surf->w, surf->h, surf->format->BytesPerPixel == 4 ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, surf->pixels);
	}

	void bindVBOs(GLhandleARB programHandle) {
		glBindAttribLocation(programHandle, vbo, "position");
	}

	void draw() {
		glUseProgram(shaderProgram);
		int projectionMatrixLocation = glGetUniformLocation(shaderProgram, "projectionMatrix");
		int viewMatrixLocation = glGetUniformLocation(shaderProgram, "viewMatrix");
		int modelMatrixLocation = glGetUniformLocation(shaderProgram, "modelMatrix");
		glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]); 
		glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]); 
		glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]); 

		int location = glGetUniformLocationARB(shaderProgram, "tex");
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glUniform1iARB(location, 0);

		glBindVertexArray(vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray( vbo );

		glVertexAttribPointer(vbo, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
		glDrawArrays( GL_TRIANGLE_STRIP, 0, numVertices );
	}

};
