//====================================================================================================100
//		UPDATE
//====================================================================================================100

//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments

//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "define.c"
#include "graphics.c"
#include "resize.c"
#include "timer.c"
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}
//====================================================================================================100
//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100
//====================================================================================================100

#define TEXTURE_1D_SIZE 131072
#define CONSTANT_SIZE 65536

#ifdef IS_CONST
__constant__ float iS_const[CONSTANT_SIZE-100];
#endif

#ifdef IMAGE_MAP
#define IMAGE_UNI
#endif

#ifdef IMAGE_DEF
#define IMAGE_PIN
#endif

#ifdef C_TEX
texture<float, 1, cudaReadModeElementType> c_texture;
#endif

#ifdef IN_TEX
texture<int, 1, cudaReadModeElementType> iN_texture;
#endif
#ifdef IS_TEX
texture<int, 1, cudaReadModeElementType> iS_texture;
#endif
#ifdef JW_TEX
texture<int, 1, cudaReadModeElementType> jW_texture;
#endif
#ifdef JE_TEX
texture<int, 1, cudaReadModeElementType> jE_texture;
#endif


__global__ void Kernel1(float *image, float *c, float *dN, float *dS, float *dW, float *dE, int *iN, int *iS, int *jW, int *jE, long Nr, float q0sqr, long Nc) {
            long j = blockIdx.x*blockDim.x + threadIdx.x;
            long i = blockIdx.y*blockDim.y + threadIdx.y;
            if (j >= Nc || i >= Nr) {
                return;
            }
                // current index/pixel
                long k = i + Nr*j;												// get position of current element
                float Jc = image[k];													// get value of the current element
                // directional derivates (every element of IMAGE)
#ifdef IN_TEX
                dN[k] = image[tex1Dfetch(iN_texture,i) + Nr*j] - Jc;								// north direction derivative
#else
                dN[k] = image[iN[i] + Nr*j] - Jc;								// north direction derivative
#endif
#ifdef IS_TEX
                dS[k] = image[tex1Dfetch(iS_texture,i) + Nr*j] - Jc;								// south direction derivative
#else
                dS[k] = image[iS[i] + Nr*j] - Jc;								// south direction derivative
#endif
#ifdef JW_TEX
                dW[k] = image[i + Nr*tex1Dfetch(jW_texture,j)] - Jc;								// west direction derivative
#else
                dW[k] = image[i + Nr*jW[j]] - Jc;								// west direction derivative
#endif
#ifdef JE_TEX
                dE[k] = image[i + Nr*tex1Dfetch(jE_texture,j)] - Jc;								// east direction derivative
#else
                dE[k] = image[i + Nr*jE[j]] - Jc;								// east direction derivative
#endif
                // normalized discrete gradient mag squared (equ 52,53)
                float G2 = (dN[k]*dN[k] + dS[k]*dS[k]								// gradient (based on derivatives)
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);
                // normalized discrete laplacian (equ 54)
                float L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;					// laplacian (based on derivatives)
                // ICOV (equ 31/35)
                float num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;						// num (based on gradient and laplacian)
                float den  = 1 + (.25*L);											// den (based on laplacian)
                float qsqr = num/(den*den);										// qsqr (based on num and den)
                // diffusion coefficent (equ 33) (every element of IMAGE)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;					// den (based on qsqr and q0sqr)
                c[k] = 1.0 / (1.0+den) ;									// diffusion coefficient (based on den)
                // saturate diffusion coefficent to 0-1 range
                if (c[k] < 0)												// if diffusion coefficient < 0
					{c[k] = 0;}												// ... set to 0
                else if (c[k] > 1)											// if diffusion coefficient > 1
					{c[k] = 1;}												// ... set to 1
        }

        __global__ void Kernel2(float *image, float *c, float *dN, float *dS, float *dW, float *dE, int * iS, int * jE, long Nr,long Nc, float lambda) {
            long j = blockIdx.x*blockDim.x + threadIdx.x;
            long i = blockIdx.y*blockDim.y + threadIdx.y;
                long k = i + Nr*j;												// get position of current element
            if (j >= Nc || i >= Nr) {
                return;
            }
                // current index
#ifdef IS_TEX
            int iS_index = tex1Dfetch(iS_texture,i);
#else
            int iS_index = iS[i];
#endif
#ifdef JE_TEX
            int jE_index = tex1Dfetch(jE_texture,j);
#else
            int jE_index = jE[j];
#endif
                // diffusion coefficent
#ifdef C_TEX
                float cN = tex1Dfetch(c_texture,k);													// north diffusion coefficient
                float cS = tex1Dfetch(c_texture,iS_index + Nr*j);										// south diffusion coefficient
                float cW = tex1Dfetch(c_texture,k);													// west diffusion coefficient
                float cE = tex1Dfetch(c_texture,i + Nr*jE_index);										// east diffusion coefficient
#else
                float cN = c[k];													// north diffusion coefficient
                float cS = c[iS_index + Nr*j];										// south diffusion coefficient
                float cW = c[k];													// west diffusion coefficient
                float cE = c[i + Nr*jE_index];										// east diffusion coefficient
#endif
                // divergence (equ 58)
                float D = cN*dN[k] + cS*dS[k] + cW*dW[k] + cE*dE[k];				// divergence

                // image update (equ 61) (every element of IMAGE)
                image[k] = image[k] + 0.25*lambda*D;								// updates image (based on input time step and divergence)
            }

int main(int argc, char *argv []){

	//================================================================================80
	// 	VARIABLES
	//================================================================================80

	// time
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;
	long long time7;
	long long time8;
	long long time9;
	long long time10;

	time0 = get_time();

    // inputs image, input paramenters
    fp* image_ori;																// originalinput image
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

    // inputs image, input paramenters
    fp* image;															// input image
    long Nr,Nc;													// IMAGE nbr of rows/cols/elements
	long Ne;

	// algorithm parameters
    int niter;																// nbr of iterations
    fp lambda;															// update step size

    // size of IMAGE
	int r1,r2,c1,c2;												// row/col coordinates of uniform ROI
	long NeROI;														// ROI nbr of elements
    
    // ROI statistics
    fp meanROI, varROI, q0sqr;											//local region statistics
    
    // surrounding pixel indicies
    int *iN,*iS,*jE,*jW;    

    // center pixel value
    fp Jc;

	// directional derivatives
	fp *dN,*dS,*dW,*dE;
    
    // calculation variables
    fp tmp,sum,sum2;
    fp G2,L,num,den,qsqr,D;
       
    // diffusion coefficient
    fp *c; 
	fp cN,cS,cW,cE;
    
    // counters
    int iter;   // primary loop
    long i,j;    // image row/col
    long k;      // image single index    

	// number of threads
	int threads;

    char *file;

	time1 = get_time();

	//================================================================================80
	// 	GET INPUT PARAMETERS
	//================================================================================80

	if(argc != 7){
		printf("ERROR: wrong number of arguments\n");
		return 0;
	}
	else{
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		Nr = atoi(argv[3]);						// it is 502 in the original image
		Nc = atoi(argv[4]);						// it is 458 in the original image
		threads = atoi(argv[5]);
		file = (argv[6]);
	}

	omp_set_num_threads(threads);
	// printf("THREAD %d\n", omp_get_thread_num());
	// printf("NUMBER OF THREADS: %d\n", omp_get_num_threads());

	time2 = get_time();

	//================================================================================80
	// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	//================================================================================80

    // read image
	image_ori_rows = 502;
	image_ori_cols = 458;
	image_ori_elem = image_ori_rows * image_ori_cols;

	image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

	read_graphics(	file,
								image_ori,
								image_ori_rows,
								image_ori_cols,
								1);

	time3 = get_time();

	//================================================================================80
	// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	//================================================================================80

	Ne = Nr*Nc;

#if defined IMAGE_DEF
    CudaSafeCall(cudaHostAlloc(&image, sizeof(float) * Ne, cudaHostAllocDefault));
#elif defined IMAGE_PIN
    CudaSafeCall(cudaMallocHost(&image, sizeof(float) * Ne));
#elif defined IMAGE_MAP
    CudaSafeCall(cudaHostAlloc(&image, sizeof(float) * Ne, cudaHostAllocMapped));
#elif defined IMAGE_UNI
    cudaMallocManaged(&image, sizeof(float) * Ne);
#else
	image = (fp*)malloc(sizeof(fp) * Ne);
#endif
	resize(	image_ori,
				image_ori_rows,
				image_ori_cols,
				image,
				Nr,
				Nc,
				1);
	time4 = get_time();

	//================================================================================80
	// 	SETUP
	//================================================================================80

    r1     = 0;											// top row index of ROI
    r2     = Nr - 1;									// bottom row index of ROI
    c1     = 0;											// left column index of ROI
    c2     = Nc - 1;									// right column index of ROI

    // ROI image size    
    NeROI = (r2-r1+1)*(c2-c1+1);											// number of elements in ROI, ROI size
    
    // allocate variables for surrounding pixels
    iN = (int*) malloc(sizeof(int*)*Nr) ;									// north surrounding element
    iS = (int*)malloc(sizeof(int*)*Nr) ;									// south surrounding element
    jW = (int*)malloc(sizeof(int*)*Nc) ;									// west surrounding element
    jE = (int*)malloc(sizeof(int*)*Nc) ;									// east surrounding element
    
	// allocate variables for directional derivatives
	dN = (float*)malloc(sizeof(fp)*Ne) ;											// north direction derivative
    dS = (float*)malloc(sizeof(fp)*Ne) ;											// south direction derivative
    dW = (float*)malloc(sizeof(fp)*Ne) ;											// west direction derivative
    dE = (float*)malloc(sizeof(fp)*Ne) ;											// east direction derivative

	// allocate variable for diffusion coefficient
#ifdef C_DEVICE_PIN
    CudaSafeCall(cudaMallocHost(&c, sizeof(float) * Ne));
#else
    c  = (float*)malloc(sizeof(fp)*Ne) ;											// diffusion coefficient
#endif

    // N/S/W/E indices of surrounding pixels (every element of IMAGE)
	// #pragma omp parallel
    for (i=0; i<Nr; i++) {
        iN[i] = i-1;														// holds index of IMAGE row above
        iS[i] = i+1;														// holds index of IMAGE row below
    }
	// #pragma omp parallel
    for (j=0; j<Nc; j++) {
        jW[j] = j-1;														// holds index of IMAGE column on the left
        jE[j] = j+1;														// holds index of IMAGE column on the right
    }
	// N/S/W/E boundary conditions, fix surrounding indices outside boundary of IMAGE
    iN[0]    = 0;															// changes IMAGE top row index from -1 to 0
    iS[Nr-1] = Nr-1;														// changes IMAGE bottom row index from Nr to Nr-1 
    jW[0]    = 0;															// changes IMAGE leftmost column index from -1 to 0
    jE[Nc-1] = Nc-1;														// changes IMAGE rightmost column index from Nc to Nc-1

	time5 = get_time();

	//================================================================================80
	// 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//================================================================================80

	// #pragma omp parallel
	for (i=0; i<Ne; i++) {													// do for the number of elements in input IMAGE
		image[i] = exp(image[i]/255);											// exponentiate input IMAGE and copy to output image
    }

	time6 = get_time();

	//================================================================================80
	// 	COMPUTATION
	//================================================================================80

	// printf("iterations: ");

    // primary loop
        printf("Ne size: %ld\n", Ne);
        float *image_d, *c_d, *dN_d, *dS_d, *dE_d, *dW_d;
        int *iN_d, *iS_d, *jW_d, *jE_d;
#ifdef IMAGE_UNI
#ifdef IMAGE_MAP
        cudaHostGetDevicePointer(&image_d, image,0);
#else
        image_d = image;
#endif
#else
        cudaMalloc(&image_d, sizeof(float) * Ne);
#endif
        cudaMalloc(&c_d, sizeof(float) * Ne);
        cudaMalloc(&dN_d, sizeof(float) * Ne);
        cudaMalloc(&dS_d, sizeof(float) * Ne);
        cudaMalloc(&dE_d, sizeof(float) * Ne);
        cudaMalloc(&dW_d, sizeof(float) * Ne);
        cudaMalloc(&iN_d, sizeof(int) * Ne);
        cudaMalloc(&iS_d, sizeof(int) * Ne);
        cudaMalloc(&jW_d, sizeof(int) * Ne);
        cudaMalloc(&jE_d, sizeof(int) * Ne);
#ifdef C_TEX
        if (sizeof(float)*Ne > TEXTURE_1D_SIZE) {
            fprintf(stderr, "Array 'c' size: %d bigger than texture size %d\n", sizeof(float)*Ne, TEXTURE_1D_SIZE);
            //return -1;
        }
        CudaSafeCall(cudaBindTexture(0, c_texture, c_d, sizeof(float) * Ne));
#endif
#ifdef IN_TEX
        CudaSafeCall(cudaBindTexture(0, iN_texture, iN_d, sizeof(int) * Ne));
#endif
#ifdef IS_TEX
        CudaSafeCall(cudaBindTexture(0, iS_texture, iS_d, sizeof(int) * Ne));
#endif

#ifdef JW_TEX
        CudaSafeCall(cudaBindTexture(0, jW_texture, jW_d, sizeof(int) * Ne));
#endif
#ifdef JE_TEX
        CudaSafeCall(cudaBindTexture(0, jE_texture, jE_d, sizeof(int) * Ne));
#endif

        cudaMemcpy(iN_d, iN, Ne*sizeof(int), cudaMemcpyHostToDevice);
#if defined IS_CONST
        CudaSafeCall(cudaMemcpyToSymbol(iS_const, iS, Ne*sizeof(int)));
#else
        cudaMemcpy(iS_d, iS, Ne*sizeof(int), cudaMemcpyHostToDevice);
#endif
        cudaMemcpy(jW_d, jW, Ne*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(jE_d, jE, Ne*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(c_d, c, Ne*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dN_d, dN, Ne*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dS_d, dS, Ne*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dE_d, dE, Ne*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dW_d, dW, Ne*sizeof(float), cudaMemcpyHostToDevice);
        dim3 grid((Nc+31)/32,(Nr+31)/32);
        dim3 block(32, 32);
        printf("X: %d, Y: %d, Z: %d\n", grid.x, grid.y, grid.z); 
    
        for (iter=0; iter<niter; iter++){										// do for the number of iterations input parameter

		// printf("%d ", iter);
		// fflush(NULL);

        // ROI statistics for entire ROI (single number for ROI)
        sum=0; 
		sum2=0;
        for (i=r1; i<=r2; i++) {											// do for the range of rows in ROI
            for (j=c1; j<=c2; j++) {										// do for the range of columns in ROI
                tmp   = image[i + Nr*j];										// get coresponding value in IMAGE
                sum  += tmp ;												// take corresponding value and add to sum
                sum2 += tmp*tmp;											// take square of corresponding value and add to sum2
            }
        }
        meanROI = sum / NeROI;												// gets mean (average) value of element in ROI
        varROI  = (sum2 / NeROI) - meanROI*meanROI;							// gets variance of ROI
        q0sqr   = varROI / (meanROI*meanROI);								// gets standard deviation of ROI
#ifndef IMAGE_UNI
        cudaMemcpy(image_d, image, Ne*sizeof(float), cudaMemcpyHostToDevice);
#endif
        Kernel1<<<grid,block>>>(image_d, c_d, dN_d, dS_d, dW_d, dE_d, iN_d, iS_d, jW_d, jE_d, Nr, q0sqr ,Nc);
        CudaCheckError();
        Kernel2<<<grid,block>>>(image_d, c_d, dN_d, dS_d, dW_d, dE_d, iS_d, jE_d, Nr, Nc, lambda);
        CudaCheckError();
#ifndef IMAGE_UNI
        cudaMemcpy(image, image_d, Ne*sizeof(float), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);
#endif
	}

	// printf("\n");

	time7 = get_time();

	//================================================================================80
	// 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//================================================================================80

	// #pragma omp parallel
	for (i=0; i<Ne; i++) {													// do for the number of elements in IMAGE
		image[i] = log(image[i])*255;													// take logarithm of image, log compress
	}

	time8 = get_time();

	//================================================================================80
	// 	WRITE IMAGE AFTER PROCESSING
	//================================================================================80

	write_graphics(	(char *)"image_out.pgm",
								image,
								Nr,
								Nc,
								1,
								255);

	time9 = get_time();

	//================================================================================80
	// 	DEALLOCATE
	//================================================================================80

	free(image_ori);
#ifdef IMAGE_PIN
    cudaFreeHost(image);
#elif defined IMAGE_UNI
    cudaFree(image);
#else 
    free(image);
#endif

    free(iN); free(iS); free(jW); free(jE);									// deallocate surrounding pixel memory
    free(dN); free(dS); free(dW); free(dE);									// deallocate directional derivative memory
#ifdef C_DEVICE_PIN
    cudaFreeHost(c);
#else
    free(c);			
#endif    // deallocate diffusion coefficient memory

	time10 = get_time();

	//================================================================================80
	//		DISPLAY TIMING
	//================================================================================80

	printf("Time spent in different stages of the application:\n");
	printf("%.12f s, %.12f % : SETUP VARIABLES\n", 									(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : READ COMMAND LINE PARAMETERS\n", 	(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : READ IMAGE FROM FILE\n", 						(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : RESIZE IMAGE\n", 										(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : SETUP, MEMORY ALLOCATION\n", 				(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : EXTRACT IMAGE\n", 									(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : COMPUTE\n", 												(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : COMPRESS IMAGE\n", 									(float) (time8-time7) / 1000000, (float) (time8-time7) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : SAVE IMAGE INTO FILE\n", 							(float) (time9-time8) / 1000000, (float) (time9-time8) / (float) (time10-time0) * 100);
	printf("%.12f s, %.12f % : FREE MEMORY\n", 										(float) (time10-time9) / 1000000, (float) (time10-time9) / (float) (time10-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n", 																					(float) (time10-time0) / 1000000);

//====================================================================================================100
//	END OF FILE
//====================================================================================================100

}


