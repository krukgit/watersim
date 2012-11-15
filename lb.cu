#ifndef _SIMPLEGL_KERNEL_H_
#define _SIMPLEGL_KERNEL_H_
#define v 2.0		// velocity
__device__ __constant__ int dx[] = {0, -1, -1, 0, 1, 1, 1, 0, -1};
__device__ __constant__ int dy[] = {0, 0, 1, 1, 1, 0, -1, -1, -1, 0};
__device__ __constant__ float w[] = {4./9., 1./9., 1./36., 1./9., 1./36., 1./9., 1./36., 1./9., 1./36.}; 
__device__ __constant__ float v_x[] = {0, v, v, 0, -v, -v, -v, 0, v};
__device__ __constant__ float v_y[] = {0, 0, v, v, v, 0, -v, -v, -v};
__device__ __constant__ float cs2 = v*v/3.;
__device__ __constant__ float tau = 0.8;


__global__ void stepKernel(unsigned int width, float *gfin, float* gfout, float fx, float fy)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int offset	= y*width*9 + x*9;
	
	float* fin	= gfin + offset;
	float* fout = gfout + offset;

	if (fout[0] < 0.0)
		return;

	float feq[9];

	float u_x	= tau * fx;
	float u_y	= tau * fy;
	float p		= 0.0;
	
	for (int k=0; k<9; k++)
		p += fin[k];

	for (int k=0; k<9; k++)
	{
		u_x += v_x[k] * fin[k] / p;
		u_y += v_y[k] * fin[k] / p;
	}
	for (int k=0; k<9; k++)
		feq[k] =  p * w[k] * (1. + 1./cs2*(v_x[k]*u_x + v_y[k]*u_y) + 1./2./cs2/cs2*((v_x[k]*v_x[k] - cs2)*u_x*u_x + 2.*v_x[k]*v_y[k]*u_x*u_y + (v_y[k]*v_y[k] - cs2)*u_y*u_y));
	
	for (int k=0; k<9; k++)
		fout[k] = fin[k] + (feq[k] - fin[k])/tau;
}

__global__ void proKernel(float4* pos, unsigned int width, unsigned int height, float *gfin, float* gfout)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
	float uu = (float)x / (float)width;
	float vv = (float)y / (float)height;

	uu = uu * 2.0f - 1.0f;
	vv = vv * 2.0f - 1.0f;

	int offset = x + y * width;
	
	float* fin = gfin + offset * 9;
	float* fout = gfout + offset * 9;
	
	if (fout[0] < -999.0) {
		pos[offset] = make_float4(uu,-100.0f,vv,1.0f);
		pos[width*height + offset] = make_float4(0.,0.,0.,1.0);
		return;
	}

	fin[0] = fout[0];
	for (int k=1; k<9; k++)
	{
		int mx = x + dx[k];
		int my = y + dy[k];
		int mk = (k + 3) % 8 + 1;

		if (mx < 0 || mx >= width || my < 0 || my >= height || gfout[my * width * 9 + mx * 9 + k] < -999.0) 
			fin[k] = fout[mk];
		else
			fin[k] = gfout[my * width * 9 + mx * 9 + k];
	}

	float h = 0.0;
	for (int i=0; i<9; i++)
		h += fin[i];
	float cScale = (h-0.9)*10.0;
	h *= cs2;	
	pos[offset] = make_float4(uu,h-1.5f,vv,1.0f);
	if (cScale > 1.0)
		cScale = 1.0;
	pos[width*height + offset] = make_float4(0.,0.5*cScale,cScale,0.0);
	//pos[width*height + offset] = make_float4(uu,vv,0.0,0.0);
}

__global__ void addDropKernel(unsigned int width, float *gfin, int i, int j, int r)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	float* fin = gfin + offset * 9;

	if ((i-x)*(i-x) + (j-y)*(j-y) < r*r)
		fin[0] += float(r*r - (i-x)*(i-x) - (j-y)*(j-y)) / float(r*r);
}

__global__ void initKernel(unsigned int width, float *gfin, float *gfout, float ih)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	float* fin = gfin + offset * 9;
	float* fout = gfout + offset * 9;

	for (int i=0; i<9; i++)
		fin[i] = fout[i] = ih;
}

__global__ void createBlockKernel(unsigned int width, float *gfout, int x1, int y1, int x2, int y2)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * width;
	float* fout = gfout + offset * 9;
	if (x >= x1 && x <= x2 && y >= y1 && y <= y2)
	{
		if (fout[0] > 0.0)
			fout[0] = -1000.0;
		else
			fout[0] = 0.1;
	}
}

float fx = 0.0;
float fy = 0.0;
float df = 0.0001;
// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(float4* pos, unsigned int mesh_width, unsigned int mesh_height, int add, float *d_fin, float *d_fout)
{
    dim3 block(16, 16, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	
	if (add & 2) {
		int r = 3;
		int x = rand() % (mesh_width-2*r) + r;
		int y = rand() % (mesh_height-2*r) + r;
		addDropKernel<<< grid, block >>>(mesh_width, d_fin, x, y, r);
	}
	if (add & 1) 
		initKernel<<< grid, block >>>(mesh_width, d_fin, d_fout, 0.1);
	if (add & 4)
		fx += df;
	if (add & 8)
		fx -= df;
	if (add & 16)
		fy += df;
	if (add & 32)
		fx -= df;
	if (add & 64)
		fx = fy = 0.0;
	if (add & 128)
		addDropKernel<<< grid, block >>>(mesh_width, d_fin, 100, 80, 3);
	if (add & 256)
		createBlockKernel<<< grid, block >>>(mesh_width, d_fout, 40, 40, 140, 60);
	if (add & 512)
		initKernel<<< grid, block >>>(mesh_width, d_fin, d_fout, 0.11);

    stepKernel<<< grid, block >>>(mesh_width, d_fin, d_fout, fx, fy);
	proKernel<<< grid, block >>>(pos, mesh_width, mesh_height, d_fin, d_fout);
}

#endif // #ifndef _SIMPLEGL_KERNEL_H_
