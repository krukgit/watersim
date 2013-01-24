#ifndef GLPROGRAM_H
#define GLPROGRAM_H

#include "common.h"

class GLProgram {
protected:
	GLuint vbo;
	GLhandleARB shaderProgram;
	GLchar *fragmentShaderSource, *vertexShaderSource;
	int numVertices;	
	GLfloat *vertices;
	
public:	
	void prepare() {
		glGenBuffers( 1, &vbo ); // Generate 1 buffer
		glBindBuffer( GL_ARRAY_BUFFER, vbo );
		glBufferData( GL_ARRAY_BUFFER, numVertices*3*sizeof(GLfloat), vertices, GL_STATIC_DRAW );
	}

	void draw() {
		glUseProgram(shaderProgram);
		int projectionMatrixLocation = glGetUniformLocation(shaderProgram, "projectionMatrix");
		int viewMatrixLocation = glGetUniformLocation(shaderProgram, "viewMatrix");
		int modelMatrixLocation = glGetUniformLocation(shaderProgram, "modelMatrix");
		glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]); 
		glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]); 
		glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]); 

		//checkErrors();

		glBindVertexArray(vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray( vbo );

		glVertexAttribPointer(vbo, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, 0);
		glDrawArrays( GL_TRIANGLE_STRIP, 0, numVertices );
	}

	virtual void bindVBOs(GLhandleARB programHandle) = 0;

	

	void compileShaders() {
		GLhandleARB programHandle;
		GLint       status;
		GLint       loglen;
		GLint       maxloglen;
		GLchar     *log = 0;

        programHandle = glCreateProgram();
    
		if (programHandle < 1) {
			fprintf(stderr,"glCreateShader() failed.\n");
			getchar();
			exit(1);
		}

		glutReportErrors();

		glAttachShader(programHandle, shaderCompile(vertexShaderSource,GL_VERTEX_SHADER));
		glAttachShader(programHandle, shaderCompile(fragmentShaderSource,GL_FRAGMENT_SHADER));
    
		glutReportErrors();

		bindVBOs(programHandle);

		glLinkProgram(programHandle);
    
		glutReportErrors();

		glGetProgramiv(programHandle, GL_LINK_STATUS, &status);

		if (status != 0) {
			shaderProgram = programHandle;
			return;
		}

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
		getchar();
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

    // If status is not 0, then the source code successfully GLhandleARB
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
	getchar();
    exit(1);
}

GLchar* read(const char* file_name)
{
	std::ifstream infile (file_name,std::ifstream::binary);
	infile.seekg(0,std::ifstream::end);
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

};

#endif