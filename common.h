#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifndef COMMON_H
#define COMMON_H

#include "sdl/SDL.h"
#include "sdl/SDL_image.h"
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

#undef main

extern const unsigned int window_width, window_height, mesh_width, mesh_height;

extern glm::mat4 projectionMatrix; // Store the projection matrix  
extern glm::mat4 viewMatrix; // Store the view matrix  
extern glm::mat4 modelMatrix; // Store the model matrix  

extern int mouse_old_x, mouse_old_y, mouse_buttons;
extern float rotate_x, rotate_y, translate_x, translate_y, translate_z;

#endif