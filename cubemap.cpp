#include "common.h"
#include "GLProgram.h"
#include "cubemap.h"

const static GLushort cube_indices[] = {
                0, 1, 2, 3,
                3, 2, 6, 7,
                7, 6, 5, 4,
                4, 5, 1, 0,
                0, 3, 7, 4,
                1, 2, 6, 5,
        };
const float a = 10.0;
const float b = -10.5;
const float t = 2*a+b;
		GLfloat cube_vertices[] = {
			-a,	t,	a,
			-a,	b,	a,
			a,	b,	a,
			a,	t,	a,
			-a,	t,	-a,
			-a,	b,	-a,
			a,	b,	-a,
			a,	t,	-a
		};

	cubemapProgram::cubemapProgram() {
		numVertices = 8;
		numIndices = 6;

		vertexShaderSource = read("cubemap.vs");
		fragmentShaderSource = read("cubemap.fs");
	}


	void cubemapProgram::prepare() {
		printf("Preparing cubemap\n");
		SDL_Surface *xpos = IMG_Load("media/interstellar_ft.tga"); SDL_Surface *xneg = IMG_Load("media/interstellar_bk.tga");
		SDL_Surface *ypos = IMG_Load("media/interstellar_up.tga"); SDL_Surface *yneg = IMG_Load("media/interstellar_dn.tga");
		SDL_Surface *zpos = IMG_Load("media/interstellar_rt.tga"); SDL_Surface *zneg = IMG_Load("media/interstellar_lf.tga");

		setupCubeMap(cubemap_texture, xpos, xneg, ypos, yneg, zpos, zneg);
		SDL_FreeSurface(xneg);  SDL_FreeSurface(xpos);
		SDL_FreeSurface(yneg);  SDL_FreeSurface(ypos);
		SDL_FreeSurface(zneg);  SDL_FreeSurface(zpos);
		
		GLint vertex = glGetAttribLocation(shaderProgram, "vertex");

		// cube vertices for vertex buffer object
		glGenBuffers(1, &vbo_cube_vertices);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_vertices);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
		//glBufferData(GL_ARRAY_BUFFER, numVertices*3*sizeof(GLfloat), 0, GL_STATIC_DRAW);
		
		//glBindBuffer(GL_ARRAY_BUFFER, 0);

		// cube indices for index buffer object
		glGenBuffers(1, &ibo_cube_indices);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_cube_indices);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);

		//glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices*4*sizeof(GLushort), 0, GL_STATIC_DRAW);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			       
		glEnableVertexAttribArray(vbo_cube_vertices);
		glVertexAttribPointer(vbo_cube_vertices, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), 0);
		//glVertexAttribPointer(vertex, 
		
	}

	void cubemapProgram::bindVBOs(GLhandleARB programHandle) {
		glBindAttribLocation(programHandle, vbo_cube_vertices, "vertex");
	}

	void cubemapProgram::draw() {// grab the pvm matrix and vertex location from our shader program

		glUseProgram(shaderProgram);
	
		GLint PVM    = glGetUniformLocation(shaderProgram, "PVM");
		GLint vertex = glGetAttribLocation(shaderProgram, "vertex");

		glm::mat4 M = projectionMatrix * viewMatrix * modelMatrix;
		glUniformMatrix4fv(PVM, 1, GL_FALSE, &M[0][0]);

		glBindVertexArray(vbo_cube_vertices);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_cube_indices);

		glDrawElements(GL_QUADS, sizeof(cube_indices)/sizeof(GLushort), GL_UNSIGNED_SHORT, 0);
		//glLoadIdentity();

//		glBindVertexArray(ibo_cube_indices);
	//	glBindBuffer(GL_ARRAY_BUFFER, ibo_cube_indices);

		//glBindVertexArray(vbo_cube_vertices);
		//glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_vertices);

/*		glBindVertexArray(vbo_cube_vertices);
				glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_vertices);
		glEnableVertexAttribArray(vbo_cube_vertices);
			//glBufferData(GL_ARRAY_BUFFER, numVertices*3*sizeof(GLfloat), cube_vertices, GL_STATIC_DRAW);


			glBindVertexArray(ibo_cube_indices);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_cube_indices);
			//glEnableVertexAttribArray(ibo_cube_vertices);
		//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		*/
//	glEnableVertexAttribArray(vertex);
	//glVertexAttribPointer(vertex, 3, GL_FLOAT, GL_FALSE, 0, 0);
}


	void cubemapProgram::setupCubeMap(GLuint& texture) {
		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_CUBE_MAP);
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	}
 
	void cubemapProgram::setupCubeMap(GLuint& texture, SDL_Surface *xpos, SDL_Surface *xneg, SDL_Surface *ypos, SDL_Surface *yneg, SDL_Surface *zpos, SDL_Surface *zneg) {
		setupCubeMap(texture);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA, xpos->w, xpos->h, 0, xpos->format->BytesPerPixel == 4 ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, xpos->pixels);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA, xneg->w, xneg->h, 0, xneg->format->BytesPerPixel == 4 ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, xneg->pixels);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA, ypos->w, ypos->h, 0, ypos->format->BytesPerPixel == 4 ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, ypos->pixels);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA, yneg->w, yneg->h, 0, yneg->format->BytesPerPixel == 4 ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, yneg->pixels);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA, zpos->w, zpos->h, 0, zpos->format->BytesPerPixel == 4 ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, zpos->pixels);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA, zneg->w, zneg->h, 0, zneg->format->BytesPerPixel == 4 ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, zneg->pixels);
	}
 
	void cubemapProgram::deleteCubeMap(GLuint& texture) {
		glDeleteTextures(1, &texture);
	}
