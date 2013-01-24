#ifndef CUBEMAP_H
#define CUBEMAP_H
#include "common.h"
#include "GLProgram.h"

class cubemapProgram : public GLProgram {
	GLuint cubemap_texture;
	GLuint vbo_cube_vertices;
	GLuint ibo_cube_indices;
	int numIndices;
public:
	cubemapProgram();
	void prepare();
	void bindVBOs(GLhandleARB programHandle);

	void draw();
	void setupCubeMap(GLuint& texture);
	void setupCubeMap(GLuint& texture, SDL_Surface *xpos, SDL_Surface *xneg, SDL_Surface *ypos, SDL_Surface *yneg, SDL_Surface *zpos, SDL_Surface *zneg);
	void deleteCubeMap(GLuint& texture);

};

#endif