/*	+++++INCLUDEs+++++	*/
// C
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>        	/* errno */
#include <string.h>       	/* strerror */
#include <math.h>			// ceil
#include <time.h>			// CLOCKS_PER_SEC
#include <inttypes.h>
// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
// other includes
#include "/home/giuliano/git/cuda/weatherprog-cudac/includes/gis.h"
//#include "/home/giuliano/git/cuda/weatherprog-cudac/includes/gis_proc.c"
/*	+++++INCLUDEs+++++	*/


/*	+++++DEFINEs+++++	*/
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
/*	+++++DEFINEs+++++	*/


/* ************* GLOBAL VARs ************* */
//	input:
unsigned int		Nmaps	= 2;
// 	---normal case study---
//const char			*iDIR 	= "/home/giuliano/work/Projects/LIFE_Project/run_clime_daily/run#2/maps/";
// 	---large case study---
const char			*iDIR 	= "/home/giuliano/git/cuda/weatherprog-cudac/data/";
const char			*clPAR	= "temp_min_h24";
const char			*YEAR	= "2011";
const char			*MONTH	= "01";
const char			*SPATMOD= "gprism";
const char			*RES	= "80";
const char			*EXT	= ".tif";
//	output:
const char			*oDIR	= "/home/giuliano/git/cuda/weatherprog-cudac/data/";
const char 			*oFILc 	= "sum_C.tif";
const char 			*oFILcu	= "sum_CUDA.tif";
// ************* GLOBAL VARs ************* //

__global__ void reduction_3d_sum( const double *lin_maps, unsigned int map_len, unsigned int Nmaps, double *sum_map ){
	/*
	 * 		lin_maps:	|------|------|------|	...	|------|
	 * 					   1st	  2nd	 3rd	...	   Nth
	 *
	 * 		block:		blockDim.Y(=1) * blockDim.X(=32^2, other);
	 * 		grid:		gridDim.Y(=1)  * gridDim.X(=map_len / blockDim.X);
	 */
	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int gdx		= gridDim.x;
	//unsigned int gdy		= gridDim.y;
	unsigned long int tid 	= (  c + bdx*( r + bdy * (bix + gdx*biy) )  );
/* 			  				     |_________|
 * 								 block-offset
 * 										     + |_____grid-offset_____|
 */
	unsigned int ii 		= 0;
	for( ii=0; ii<Nmaps-(Nmaps % 2); ii+=2 ){
		if(tid < map_len){
			sum_map[tid] = sum_map[tid] + lin_maps[tid + ii*map_len] + lin_maps[(ii+1)*map_len + tid];
		}
	}
	if(Nmaps % 2){
		if(tid < map_len){
			sum_map[tid] = sum_map[tid] + lin_maps[(Nmaps-1)*map_len + tid];
		}
	}
}


int main(int argc, char **argv){

	metadata 		MD;
	double			iMap_bytes;
	unsigned int	map_len;
	char			DAY[8];
	unsigned int	ii=0;
	// clocks:
	clock_t			start_C,end_C,start_CU,end_CU;

	/*
	 * 		ESTABILISH CONTEXT
	 */
	GDALAllRegister();	// Establish GDAL context.
	cudaFree(0); 		// Establish CUDA context.

	/*
	 * 		DEFINE I/O files
	 */
	char iFIL1[256],iFILi[256],oFIL_C[256],oFIL_CUDA[256];
	//	-I-
	snprintf(iFIL1, sizeof iFIL1, "%s%s%s%s%s%s%s%s%s%s%s", iDIR, clPAR, "-", YEAR, MONTH, "01", "-", SPATMOD, "-", RES, EXT);
	//	-O-
	snprintf(oFIL_C, 	sizeof oFIL_C, 		"%s%s", oDIR, oFILc);
	snprintf(oFIL_CUDA, sizeof oFIL_CUDA, 	"%s%s", oDIR, oFILcu);

	/*
	 * 		IMPORT METADATA
	 */
	MD 				= geotiffinfo( iFIL1, 1 );
	map_len 		= MD.width*MD.heigth;

/*	switch(MD.pixel_type){
		case GDT_Float64:
			iMap_bytes			= map_len*sizeof( double );
			double *iGRIDi		= (double *) CPLMalloc( iMap_bytes*Nmaps );
			double *oGRID		= (double *) CPLMalloc( iMap_bytes );
			break;
		case GDT_Float32:
			iMap_bytes			= MD.width*MD.heigth*sizeof( float );
			float *iGRID1		= (float *) CPLMalloc( iMap_bytes );
			float *iGRID2		= (float *) CPLMalloc( iMap_bytes );
			float *oGRID		= (float *) CPLMalloc( iMap_bytes );
			break;
		case GDT_UInt32:
			iMap_bytes			= MD.width*MD.heigth*sizeof( uint32_t );
			uint32_t *iGRID1	= (uint32_t *) CPLMalloc( iMap_bytes );
			uint32_t *iGRID2	= (uint32_t *) CPLMalloc( iMap_bytes );
			uint32_t *oGRID		= (uint32_t *) CPLMalloc( iMap_bytes );
			break;
		case GDT_Int32:
			iMap_bytes			= MD.width*MD.heigth*sizeof( int32_t );
			int32_t *iGRID1		= (int32_t *) CPLMalloc( iMap_bytes );
			int32_t *iGRID2		= (int32_t *) CPLMalloc( iMap_bytes );
			int32_t *oGRID		= (int32_t *) CPLMalloc( iMap_bytes );
			break;
		default:
			printf("Error: the current gdal data type is not yet implemented!\n");
			exit(EXIT_FAILURE);
	}
*/

	iMap_bytes			= map_len*sizeof( double );
	double *iGRIDi		= (double *) CPLMalloc( iMap_bytes*Nmaps );
	double *oGRID		= (double *) CPLMalloc( iMap_bytes );

	// import Map using GRID-filename:
	for(ii=0;ii<Nmaps;ii++){
		if(ii<9){ snprintf(DAY,sizeof DAY,"0%d",ii+1); }
		else{snprintf(DAY,sizeof DAY,"%d",ii+1);}
		//printf("DAY:\t%s\n",DAY);
		snprintf(iFILi, sizeof iFILi, "%s%s%s%s%s%s%s%s%s%s%s", iDIR, clPAR, "-", YEAR, MONTH, DAY, "-", SPATMOD, "-", RES, EXT);
		geotiffread( iFILi, MD, &iGRIDi[0] + ii*map_len );
	}


	/*	C sum	*/
	uint32_t loc;//rr,cc,

	// initialize oMap:
	for( loc=0; loc<map_len; loc++ ) oGRID[loc]=0;

	start_C = clock();
	for( ii=0; ii<Nmaps-(Nmaps % 2); ii+=2 ){
		for( loc=0; loc<map_len; loc++ ) oGRID[loc] = oGRID[loc] + iGRIDi[ii*map_len + loc] + iGRIDi[(ii+1)*map_len + loc];
	}
	if(Nmaps % 2) for( loc=0; loc<map_len; loc++ ) oGRID[loc] = oGRID[loc] + iGRIDi[(Nmaps-1)*map_len + loc];
	geotiffwrite( iFIL1, oFIL_C, MD, oGRID );
	end_C = clock();
	printf("Total time: %f [msec]\t%s\n\n", (double)(end_C  - start_C ) / CLOCKS_PER_SEC * 1000,"-C code-" );

	/*	C sum	*/


	/*	CUDA sum	*/
	double				*dev_iGRIDi,*dev_oGRID;

	CUDA_CHECK_RETURN( cudaMalloc((void **)&dev_iGRIDi, iMap_bytes*Nmaps ) );
	CUDA_CHECK_RETURN( cudaMalloc((void **)&dev_oGRID,  iMap_bytes ) );

	CUDA_CHECK_RETURN( cudaMemcpy(dev_iGRIDi , iGRIDi,  iMap_bytes*Nmaps, cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemset(dev_oGRID, 0,  (size_t)iMap_bytes) );

	unsigned int 	BLOCKSIZE, GRIDSIZE;
	BLOCKSIZE	= 32;//floor(sqrt( devProp.maxThreadsPerBlock ));
	GRIDSIZE 	= 1 + (map_len / (BLOCKSIZE*BLOCKSIZE));
	dim3 block( BLOCKSIZE,BLOCKSIZE,1);
	dim3 grid ( GRIDSIZE,1,1);

	start_CU = clock();
	reduction_3d_sum<<<grid,block>>>( dev_iGRIDi, map_len, Nmaps, dev_oGRID );
	CUDA_CHECK_RETURN( cudaMemcpy(oGRID, dev_oGRID, iMap_bytes, cudaMemcpyDeviceToHost) );
	geotiffwrite( iFIL1, oFIL_CUDA, MD, oGRID );
	end_CU = clock();
	printf("Total time: %f [msec]\t%s\n\n", (double)(end_CU - start_CU) / CLOCKS_PER_SEC * 1000,"-CUDA code-" );
	printf( "Speedup:\t%3.2f", (double)(end_C-start_C)/(double)(end_CU-start_CU) );

	/*	CUDA sum	*/

	//------------ KILL ------------//
	/*
	 * 	FREE
	 */
	// GDAL
	VSIFree( iGRIDi );
	VSIFree( oGRID  );
	// CUDA free:
	cudaFree( dev_iGRIDi );
	cudaFree( dev_oGRID  );
	// C free:

	// Destroy context
	CUDA_CHECK_RETURN( cudaDeviceReset() );

	printf("\n\nFinished!!\n");

	return 0;
}
