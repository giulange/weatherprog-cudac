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
#include <math_functions.h>
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
//	dimension of problem:		/*	  0    		 1				*/
unsigned int		multiGPU=0;	/*	{ singleGPU, multiGPU } 	*/

//	STAT:						/*	  0    1    2    3     4		*/
unsigned int		STAT	= 3;/*	{ sum, min, max, mean, std }*/
//	input:
unsigned int		Nmaps	= 31;
// 	---SMALL---
const char			*iDIR 	= "/home/giuliano/work/Projects/LIFE_Project/run_clime_daily/run#2/maps/";
const char			*clPAR	= "rain_cum_h24";
const char			*iFIL_ROI="/home/giuliano/git/cuda/weatherprog-cudac/data/roi_vt.tif";
// 	---LARGE---
//const char			*iDIR 	= "/home/giuliano/git/cuda/weatherprog-cudac/data/";
//const char			*clPAR	= "L5_temp_min_h24";
//const char			*clPAR	= "L1_temp_min_h24";

const char			*YEAR	= "2011";
const char			*MONTH	= "01";
const char			*SPATMOD= "idw2";
const char			*RES	= "80";
const char			*EXT	= ".tif";
//	output:
const char			*oDIR	= "/home/giuliano/git/cuda/weatherprog-cudac/data/";
const char 			*oFILc 	= "out_C.tif";
const char 			*oFILcu	= "out_CUDA.tif";
const char 			*oPLOTcu= "/home/giuliano/git/cuda/weatherprog-cudac/data/oPLOTcu";
const char 			*oPLOTc = "/home/giuliano/git/cuda/weatherprog-cudac/data/oPLOTc";
// ************* GLOBAL VARs ************* //

__global__ void reduction_3d_sum( const double *lin_maps, const unsigned char *roiMap, unsigned int map_len, unsigned int Nmaps, double *sum_map ){
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
	if(tid < map_len){
		sum_map[tid] *= (double)roiMap[tid];
	}
}
__global__ void reduction_3d_min( const double *lin_maps, const unsigned char *roiMap, unsigned int map_len, unsigned int Nmaps, double *sum_map ){
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
			sum_map[tid] = fminf( fminf(sum_map[tid], lin_maps[tid + ii*map_len]) , lin_maps[(ii+1)*map_len + tid] );
		}
	}
	if(Nmaps % 2){
		if(tid < map_len){
			sum_map[tid] = fminf( sum_map[tid], lin_maps[(Nmaps-1)*map_len + tid] );
		}
	}
	if(tid < map_len){
		sum_map[tid] *= (double)roiMap[tid];
	}
}
__global__ void reduction_3d_max( const double *lin_maps, const unsigned char *roiMap, unsigned int map_len, unsigned int Nmaps, double *sum_map ){
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
			sum_map[tid] = fmaxf( fmaxf(sum_map[tid], lin_maps[tid + ii*map_len]) , lin_maps[(ii+1)*map_len + tid] );
		}
	}
	if(Nmaps % 2){
		if(tid < map_len){
			sum_map[tid] = fmaxf( sum_map[tid], lin_maps[(Nmaps-1)*map_len + tid] );
		}
	}
	if(tid < map_len){
		sum_map[tid] *= roiMap[tid];
	}
}
__global__ void reduction_3d_mean( const double *lin_maps, const unsigned char *roiMap, unsigned int map_len, unsigned int Nmaps, double *sum_map ){
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
	if(tid < map_len){
		//sum_map[tid] = (double)(sum_map[tid] * (double)roiMap[tid]) / (double)Nmaps;
		sum_map[tid] = (double)__fdividef( (float)(sum_map[tid] *roiMap[tid]) , (float)(Nmaps) );
	}
}

__global__ void reduction_2d_sum( const double *lin_maps, const unsigned char *roiMap, unsigned int map_len, unsigned int Nmaps, double *sum_map_2d ){
	/*
	 * 		lin_maps:	|------|------|------|	...	|------|
	 * 					   1st	  2nd	 3rd	...	   Nth
	 *
	 * 		block:		blockDim.Y(=32) * blockDim.X(=32);
	 * 		grid:		??;
	 */
	extern __shared__ double sdata[];

	unsigned int	tid		= threadIdx.x;
	unsigned int 	bdx		= blockDim.x;
	unsigned int 	bix		= blockIdx.x;
	unsigned int 	gdx		= gridDim.x;

	unsigned int 	nBlks 	= 0;
	unsigned int 	res		= 0;
	unsigned int 	i		= 0;

	if( bix < gdx ){

		i					= 1;
		nBlks 				= map_len / bdx;
		res					= map_len - bdx * nBlks;

		// copy the first bdx pixels to shared mem:
		sdata[tid]			= lin_maps[  tid + bdx*0 	 + bix*map_len ] *roiMap[tid];
		__syncthreads();

		/* each thread is responsible for its tid in all nBlks
		 * (excluding the residual pixels smaller than bdx):
		 */
		while( i < nBlks ){
			sdata[tid]		+= lin_maps[ tid + bdx*i	 + bix*map_len ] *roiMap[tid];
			i++;
		}
		// residual pixels not forming a block equal to bdx is handled by tid<res threads:
		if( tid<res ){
			sdata[tid]		+= lin_maps[ tid + bdx*nBlks + bix*map_len ] *roiMap[tid];
		}
		__syncthreads();

		// do reduction in shared mem of the bdx block stored in shared memory:
	    for (unsigned int s=bdx/2; s>0; s>>=1)
	    {
	        if (tid < s) sdata[tid] += sdata[tid + s];
	        __syncthreads();
	    }

	    // write result for this block to global mem:
	    if (tid == 0) sum_map_2d[bix] = sdata[0];
	}
}
__global__ void reduction_2d_mean( const double *lin_maps, const unsigned char *roiMap, unsigned int map_len, unsigned int Nmaps, double *sum_map_2d ){
	/*
	 * 		lin_maps:	|------|------|------|	...	|------|
	 * 					   1st	  2nd	 3rd	...	   Nth
	 *
	 * 		block:		blockDim.Y(=32) * blockDim.X(=32);
	 * 		grid:		??;
	 */
	extern __shared__ double sdata[];

	unsigned int	tid		= threadIdx.x;
	unsigned int 	bdx		= blockDim.x;
	unsigned int 	bix		= blockIdx.x;
	unsigned int 	gdx		= gridDim.x;

	unsigned int 	nBlks 	= 0;
	unsigned int 	res		= 0;
	unsigned int 	i		= 0;

	if( bix < gdx ){

		i					= 1;
		nBlks 				= map_len / bdx;
		res					= map_len - bdx * nBlks;

		// copy the first bdx pixels to shared mem:
		sdata[tid]			= lin_maps[  tid + bdx*0 	 + bix*map_len ] *roiMap[tid + bdx*0];
		__syncthreads();

		/* each thread is responsible for its tid in all nBlks
		 * (excluding the residual pixels smaller than bdx):
		 */
		while( i < nBlks ){
			sdata[tid]		+= lin_maps[ tid + bdx*i	 + bix*map_len ] *roiMap[tid + bdx*i];
			i++;
		}
		// residual pixels not forming a block equal to bdx is handled by tid<res threads:
		if( tid<res ){
			sdata[tid]		+= lin_maps[ tid + bdx*nBlks + bix*map_len ] *roiMap[tid + bdx*nBlks];
		}
		__syncthreads();

		// do reduction in shared mem of the bdx block stored in shared memory:
	    for (unsigned int s=bdx/2; s>0; s>>=1)
	    {
	        if (tid < s) sdata[tid] += sdata[tid + s];
	        __syncthreads();
	    }

	    // write result for this block to global mem:
	    if (tid == 0) sum_map_2d[bix] = (double)__fdividef( (float)sdata[0], (float)(map_len-1) );
	}
}
__global__ void reduction_2d_std( const double *lin_maps, const unsigned char *roiMap, unsigned int map_len, unsigned int Nmaps, double *sum_map_2d  ){
	/*
	 * 		lin_maps:	|------|------|------|	...	|------|
	 * 					   1st	  2nd	 3rd	...	   Nth
	 *
	 * 		block:		blockDim.Y(=32) * blockDim.X(=32);
	 * 		grid:		??;
	 *
	 *		Definition:
	 *		The square root of the average of the squared differences of the values from their average value
	 *
	 * 		For a given roi-ed map A, the following equation is used (A is assumed linear and not squared):
	 * 			sqrt( sum( (A-mean(A)).^2 ) / (numel(A)-1) )
	 * 		In this kernel the equation is splitted in three parts:
	 * 			(a) B = ( A-mean(A)).^2 )
	 * 				where I use "powf(...)" function
	 * 			(b) C = sum( B )
	 *				where I use "while(...)" , "if(...)" and "for(...)" statements
	 * 			(c) sqrt( C / (numel(A)-1) )
	 * 				where I use "sqrtf(...)" function
	 */
	extern __shared__ double sdata[];

	unsigned int	tid		= threadIdx.x;
	unsigned int 	bdx		= blockDim.x;
	unsigned int 	bix		= blockIdx.x;
	unsigned int 	gdx		= gridDim.x;

	unsigned int 	nBlks 	= 0;
	unsigned int 	res		= 0;
	unsigned int 	i		= 0;

	if( bix < gdx ){

		i					= 1;
		nBlks 				= map_len / bdx;
		res					= map_len - bdx * nBlks;

		// copy the first bdx pixels to shared mem:
		sdata[tid]			= powf( lin_maps[  tid + bdx*0		+ bix*map_len ] - sum_map_2d[bix], 2 ) *roiMap[tid + bdx*0];
		__syncthreads();

		/* each thread is responsible for its tid in every bix
		 * (excluding the residual pixels smaller than bdx):
		 */
		while( i < nBlks ){
			sdata[tid]		+= powf( lin_maps[ tid + bdx*i		+ bix*map_len ] - sum_map_2d[bix], 2 ) *roiMap[tid + bdx*i];
			i++;
		}
		// residual pixels not forming a block equal to bdx is handled by tid<res threads:
		if( tid<res ){
			sdata[tid]		+= powf( lin_maps[ tid + bdx*nBlks	+ bix*map_len ] - sum_map_2d[bix], 2 ) *roiMap[tid + bdx*nBlks];
		}
		__syncthreads();

		// do reduction in shared mem of the bdx block stored in shared memory:
	    for (unsigned int s=bdx/2; s>0; s>>=1)
	    {
	        if (tid < s) sdata[tid] += sdata[tid + s];
	        __syncthreads();
	    }

	    // write result for this block to global mem:
	    if (tid == 0){
	    	sum_map_2d[Nmaps*1 + bix] = __fadd_rd( sum_map_2d[bix], -(double)sqrtf(__fdividef((float)sdata[0],(double)(map_len-1))) ); // - sqrtf( __fdividef( sdata[0],(double)(map_len-1) ) );
	    	sum_map_2d[Nmaps*2 + bix] = __fadd_rd( sum_map_2d[bix], +(double)sqrtf(__fdividef((float)sdata[0],(double)(map_len-1))) ); //__fadd_rd( sum_map_2d[bix], +(double)sqrtf(__fdividef((float)sdata[0],(double)(map_len-1))) )
	    }
	}
}
__global__ void reduction_2d_ssd( const double *lin_maps, const unsigned char *roiMap, unsigned int map_len, unsigned int Nmaps, double *sum_map_2d  ){
	/*
	 * 		lin_maps:	|------|------|------|	...	|------|
	 * 					   1st	  2nd	 3rd	...	   Nth
	 *
	 * 		block:		blockDim.Y(=32) * blockDim.X(=32);
	 * 		grid:		??;
	 *
	 * 		For a given roi-ed map A, the following equation is used (A is assumed linear and not squared):
	 * 			sum( (A-mean(A)).^2 )
	 */
	extern __shared__ double sdata[];

	unsigned int	tid		= threadIdx.x;
	unsigned int 	bdx		= blockDim.x;
	unsigned int 	bix		= blockIdx.x;
	unsigned int 	gdx		= gridDim.x;

	unsigned int 	nBlks 	= 0;
	unsigned int 	res		= 0;
	unsigned int 	i		= 0;

	if( bix < gdx ){

		i					= 1;
		nBlks 				= map_len / bdx;
		res					= map_len - bdx * nBlks;

		// copy the first bdx pixels to shared mem:
		sdata[tid]			= powf( lin_maps[  tid + bdx*0		+ bix*map_len ] - sum_map_2d[bix], 2 ) *roiMap[tid + bdx*0]; // deviation of pixel from mean val
		__syncthreads();

		/* each thread is responsible for its tid in every bix
		 * (excluding the residual pixels smaller than bdx):
		 */
		while( i < nBlks ){
			sdata[tid]		+= powf( lin_maps[ tid + bdx*i		+ bix*map_len ] - sum_map_2d[bix], 2 ) *roiMap[tid + bdx*i]; // sum deviations
			i++;
		}
		// residual pixels not forming a block equal to bdx is handled by tid<res threads:
		if( tid<res ){
			sdata[tid]		+= powf( lin_maps[ tid + bdx*nBlks	+ bix*map_len ] - sum_map_2d[bix], 2 ) *roiMap[tid + bdx*nBlks]; // sum deviations
		}
		__syncthreads();

		// do reduction in shared mem of the bdx block stored in shared memory:
	    for (unsigned int s=bdx/2; s>0; s>>=1)
	    {
	        if (tid < s) sdata[tid] += sdata[tid + s];
	        __syncthreads();
	    }

	    // write result for this block to global mem:
	    if (tid == 0){
	    	sum_map_2d[Nmaps*1 + bix] = (double)sdata[0];
	    }
	}
}

int C_sum_whole_mat( double *oGRID, double *iGRIDi, unsigned char *roiMap, unsigned int map_len, char *iFIL1, char *oFIL_C, metadata MD ){

	unsigned int	ii=0;
	uint32_t 		loc;
	clock_t			start_t,end_t;

	start_t = clock();

	// initialize oGRID:
	for( loc=0; loc<map_len; loc++ ) oGRID[loc]=0;

	// computing node:
	for( ii=0; ii<Nmaps-(Nmaps % 2); ii+=2 ){
		for( loc=0; loc<map_len; loc++ ) oGRID[loc] += iGRIDi[ii*map_len + loc] + iGRIDi[(ii+1)*map_len + loc];
	}
	if(Nmaps % 2) for( loc=0; loc<map_len; loc++ ) oGRID[loc] = oGRID[loc] + iGRIDi[(Nmaps-1)*map_len + loc];

	// mask the GRID of statistics by ROI GRID:
	for( loc=0; loc<map_len; loc++ ) oGRID[loc] *= roiMap[loc];

	end_t = clock();

	// save on HDD:
	geotiffwrite( iFIL1, oFIL_C, MD, oGRID );

	// elapsed time [ms]:
	int elapsed_time = (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );
	//printf("%12s %5d [msec]\t%s\n", "Total time:",elapsed_time,"-C code-" );
	return elapsed_time;
}
int C_min_whole_mat( double *oGRID, double *iGRIDi, unsigned char *roiMap, unsigned int map_len, char *iFIL1, char *oFIL_C, metadata MD ){

	unsigned int	ii=0;
	uint32_t 		loc;
	clock_t			start_t,end_t;

	start_t = clock();

	// initialise oGRID:
	for( loc=0; loc<map_len; loc++ ) oGRID[loc]=1000;

	// computing node:
	for( ii=0; ii<Nmaps-(Nmaps % 2); ii+=2 ){
		for( loc=0; loc<map_len; loc++ ) oGRID[loc] = fminf( fminf(oGRID[loc], iGRIDi[ii*map_len + loc]) , iGRIDi[(ii+1)*map_len + loc] );
	}
	if(Nmaps % 2) for( loc=0; loc<map_len; loc++ ) oGRID[loc] = fminf( oGRID[loc], iGRIDi[(Nmaps-1)*map_len + loc] );

	// mask the GRID of statistics by ROI GRID:
	for( loc=0; loc<map_len; loc++ ) oGRID[loc] *= roiMap[loc];

	end_t = clock();

	// save on HDD:
	geotiffwrite( iFIL1, oFIL_C, MD, oGRID );

	// elapsed time [ms]:
	int elapsed_time = (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );
	//printf("%12s %5d [msec]\t%s\n", "Total time:",elapsed_time,"-C code-" );
	return elapsed_time;
}
int C_max_whole_mat( double *oGRID, double *iGRIDi, unsigned char *roiMap, unsigned int map_len, char *iFIL1, char *oFIL_C, metadata MD ){

	unsigned int	ii=0;
	uint32_t 		loc;
	clock_t			start_t,end_t;

	start_t = clock();

	// initialise oGRID:
	for( loc=0; loc<map_len; loc++ ) oGRID[loc]=-1000;

	// computing node:
	for( ii=0; ii<Nmaps-(Nmaps % 2); ii+=2 ){
		for( loc=0; loc<map_len; loc++ ) oGRID[loc] = fmaxf( fmaxf(oGRID[loc], iGRIDi[ii*map_len + loc]) , iGRIDi[(ii+1)*map_len + loc] );
	}
	if(Nmaps % 2) for( loc=0; loc<map_len; loc++ ) oGRID[loc] = fmaxf( oGRID[loc], iGRIDi[(Nmaps-1)*map_len + loc] );

	// mask the GRID of statistics by ROI GRID:
	for( loc=0; loc<map_len; loc++ ) oGRID[loc] *= roiMap[loc];

	end_t = clock();

	// save on HDD:
	geotiffwrite( iFIL1, oFIL_C, MD, oGRID );

	// elapsed time [ms]:
	int elapsed_time = (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );
	//printf("%12s %5d [msec]\t%s\n", "Total time:",elapsed_time,"-C code-" );
	return elapsed_time;
}
int C_mean_whole_mat( double *oGRID, double *iGRIDi, unsigned char *roiMap, unsigned int map_len, char *iFIL1, char *oFIL_C, metadata MD ){

	unsigned int	ii=0;
	uint32_t 		loc;
	clock_t			start_t,end_t;

	start_t = clock();

	// initialize oGRID:
	for( loc=0; loc<map_len; loc++ ) oGRID[loc]=0;

	// computing node:
	for( ii=0; ii<Nmaps-(Nmaps % 2); ii+=2 ){
		for( loc=0; loc<map_len; loc++ ) oGRID[loc] = oGRID[loc] + iGRIDi[ii*map_len + loc] + iGRIDi[(ii+1)*map_len + loc];
	}
	if(Nmaps % 2) for( loc=0; loc<map_len; loc++ ) oGRID[loc] = oGRID[loc] + iGRIDi[(Nmaps-1)*map_len + loc];

	// mask the GRID of statistics by ROI GRID:
	for( loc=0; loc<map_len; loc++ ) oGRID[loc] = oGRID[loc] * roiMap[loc] / Nmaps;

	end_t = clock();

	// save on HDD:
	geotiffwrite( iFIL1, oFIL_C, MD, oGRID );

	// elapsed time [ms]:
	int elapsed_time = (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );
	//printf("%12s %5d [msec]\t%s\n", "Total time:",elapsed_time,"-C code-" );
	return elapsed_time;
}
int C_std_whole_mat( double *oGRID, double *iGRIDi, unsigned char *roiMap, unsigned int map_len, char *iFIL1, char *oFIL_C, metadata MD ){
	/*
 	 * 		The equation is split in four parts, given population A:
 	 *
	 * 			(1) MEAN of population A			--> m = mean(A)
	 *					m:scalar, A:grid
	 * 			(2)	SQUARED DIFFERENCES				--> B = (A-m).^2
	 *
	 * 			(2) MEAN (of squared differences)	--> C = b1/(N-1) + b2/(N-1) + ... + bN/(N-1)
	 *					bi:singleton of B
	 * 			(3)	SQRT							--> STD = sqrt( C )
	 */

	double *meanGRID= (double *) CPLMalloc( map_len*sizeof( double ) );
	unsigned int	ii=0;
	uint32_t 		loc;
	clock_t			start_t,end_t;

	start_t = clock();

	// initialize oGRID:
	for( loc=0; loc<map_len; loc++ ){ oGRID[loc]=0; meanGRID[loc]=0; }

	// computing node:
	//		** -1- MEAN **
	for( ii=0; ii<Nmaps-(Nmaps % 2); ii+=2 ){
		for( loc=0; loc<map_len; loc++ ) meanGRID[loc] = meanGRID[loc] + iGRIDi[ii*map_len + loc] + iGRIDi[(ii+1)*map_len + loc];
	}
	if(Nmaps % 2) for( loc=0; loc<map_len; loc++ ) meanGRID[loc] = meanGRID[loc] + iGRIDi[(Nmaps-1)*map_len + loc];
	for( loc=0; loc<map_len; loc++ ) meanGRID[loc] = meanGRID[loc] / Nmaps;
	//		** -2+3- MEAN of SQUARED DIFF **
	for( loc=0; loc<map_len; loc++ ){
		for( ii=0; ii<Nmaps; ii++ ) oGRID[loc] += powf(iGRIDi[ii*map_len + loc]-meanGRID[loc],2) / (Nmaps-1);
		oGRID[loc] = sqrt(oGRID[loc])*roiMap[loc];
	}
	end_t = clock();
	// save on HDD:
	geotiffwrite( iFIL1, oFIL_C, MD, oGRID );

	// elapsed time [ms]:
	int elapsed_time = (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );
	//printf("%12s %5d [msec]\t%s\n", "Total time:",elapsed_time,"-C code-" );
	return elapsed_time;
}
int C_std_2D_whole_mat( double *oPLOT, double *iGRIDi, unsigned char *roiMap, unsigned int map_len, unsigned int Nmaps ){
	/*
 	 * 		The equation is split in four parts, given population A:
 	 *
	 * 			(1) MEAN of population A			--> m = mean(A)
	 *					m:scalar, A:grid
	 * 			(2)	SQUARED DIFFERENCES				--> B = (A-m).^2
	 *
	 * 			(2) MEAN (of squared differences)	--> C = b1/(N-1) + b2/(N-1) + ... + bN/(N-1)
	 *					bi:singleton of B
	 * 			(3)	SQRT							--> STD = sqrt( C )
	 */

	unsigned int	ii=0;
	uint32_t 		loc;
	clock_t			start_t,end_t;
	unsigned int	Ncols			= 3;

	start_t = clock();

	// initialize oPLOT:
	for( ii=0; ii<Nmaps*Ncols; ii++ ){ oPLOT[ii]=0; }

	// computing node:
	//		** -1- MEAN **
	for( ii=0; ii<Nmaps; ii++ ) for( loc=0; loc<map_len; loc++ ) oPLOT[ii] += iGRIDi[loc + map_len*ii]*roiMap[loc];
	for( ii=0; ii<Nmaps; ii++ ) oPLOT[ii] /= (map_len-1);
	//		** -2+3- MEAN of SQUARED DIFF **
	for( ii=0; ii<Nmaps; ii++ ) for( loc=0; loc<map_len; loc++ ) oPLOT[ii+Nmaps*1] += (powf( (iGRIDi[loc + map_len*ii]-oPLOT[ii])*roiMap[loc], 2 ) / (map_len-1));
	//		** -4- STD
	for( ii=0; ii<Nmaps; ii++ ){ oPLOT[ii+Nmaps*2] = oPLOT[ii] + sqrtf(oPLOT[ii+Nmaps*1]); oPLOT[ii+Nmaps*1] = oPLOT[ii] - sqrtf(oPLOT[ii+Nmaps*1]); }

	end_t = clock();

	// save file:
	FILE *fid = fopen(oPLOTc,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",oPLOTc); exit(1); }
	for(int rr=0;rr<Nmaps;rr++){ for(int cc=0;cc<Ncols;cc++){ fprintf(fid,"%9.5f ",oPLOT[rr+cc*Nmaps]); } fprintf(fid,"\n"); }
	fclose(fid);

	// elapsed time [ms]:
	return (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );

}
int CUDA_whole_mat( double *oGRID, double *oPLOT, double *iGRIDi, unsigned char *roiMap, unsigned int map_len, char *iFIL1, char *oFIL_CUDA, metadata MD ){

	double			*dev_iGRIDi, *dev_oGRID, *dev_oPLOT;
	unsigned char	*dev_ROI;
	clock_t			start_t,end_t;
	unsigned int 	elapsed_time	= 0;
	unsigned int	Ncols			= 3;
	double			iMap_bytes		= map_len*sizeof( double );
	double			oPLOT_bytes		= Nmaps*Ncols*sizeof( double );
	unsigned int	ii				= 0;

	// initialize grids on GPU:
	CUDA_CHECK_RETURN( cudaMalloc((void **)&dev_iGRIDi, iMap_bytes*Nmaps) 	);
	CUDA_CHECK_RETURN( cudaMalloc((void **)&dev_ROI,  	map_len*sizeof( unsigned char )) );
	CUDA_CHECK_RETURN( cudaMalloc((void **)&dev_oGRID,  iMap_bytes) 		);
	CUDA_CHECK_RETURN( cudaMalloc((void **)&dev_oPLOT,  oPLOT_bytes) 		);
	// H2D:
	CUDA_CHECK_RETURN( cudaMemcpy(dev_iGRIDi, iGRIDi, iMap_bytes*Nmaps, cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy(dev_ROI, roiMap, map_len*sizeof( unsigned char ), cudaMemcpyHostToDevice) );
	// memset:
	CUDA_CHECK_RETURN( cudaMemset(dev_oPLOT, 0,  (size_t)oPLOT_bytes) 		);

	/*
	 * 		****3-D aggregation****
	 */
	// kernel size:
	unsigned int 	BLOCKSIZE, GRIDSIZE;
	BLOCKSIZE	= 32;//floor(sqrt( devProp.maxThreadsPerBlock ));
	GRIDSIZE 	= 1 + (map_len / (BLOCKSIZE*BLOCKSIZE));
	dim3 block( BLOCKSIZE,BLOCKSIZE,1);
	dim3 grid ( GRIDSIZE,1,1);

	/*
	 *  see http://en.wikipedia.org/wiki/Double-precision_floating-point_format
	 */
	double MIN_FILL_VAL = 0x0010000000000000;
	double MAX_FILL_VAL = 0x7fefffffffffffff;

	// computing node:
	switch(STAT){
	case 0: // SUM
		CUDA_CHECK_RETURN( cudaMemset(dev_oGRID, 0,  (size_t)iMap_bytes) );
		start_t = clock();
		reduction_3d_sum<<<grid,block>>>( dev_iGRIDi, dev_ROI, map_len, Nmaps, dev_oGRID );
		CUDA_CHECK_RETURN( cudaMemcpy(oGRID, dev_oGRID, iMap_bytes, cudaMemcpyDeviceToHost) );
		end_t = clock();
		break;

	case 1: // MIN
		CUDA_CHECK_RETURN( cudaMemset(dev_oGRID, MAX_FILL_VAL,  (size_t)iMap_bytes) );
		start_t = clock();
		reduction_3d_min<<<grid,block>>>( dev_iGRIDi, dev_ROI, map_len, Nmaps, dev_oGRID );
		CUDA_CHECK_RETURN( cudaMemcpy(oGRID, dev_oGRID, iMap_bytes, cudaMemcpyDeviceToHost) );
		end_t = clock();
		break;

	case 2: // MAX
		CUDA_CHECK_RETURN( cudaMemset(dev_oGRID, MIN_FILL_VAL,  (size_t)iMap_bytes) );
		start_t = clock();
		reduction_3d_max<<<grid,block>>>( dev_iGRIDi, dev_ROI, map_len, Nmaps, dev_oGRID );
		CUDA_CHECK_RETURN( cudaMemcpy(oGRID, dev_oGRID, iMap_bytes, cudaMemcpyDeviceToHost) );
		end_t = clock();
		break;
	case 3: // MEAN
		CUDA_CHECK_RETURN( cudaMemset(dev_oGRID, 0,  (size_t)iMap_bytes) );
		start_t = clock();
		reduction_3d_mean<<<grid,block>>>( dev_iGRIDi, dev_ROI, map_len, Nmaps, dev_oGRID );
		CUDA_CHECK_RETURN( cudaMemcpy(oGRID, dev_oGRID, iMap_bytes, cudaMemcpyDeviceToHost) );
		end_t = clock();
		break;
	}
	// elapsed time [ms]:
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );

	/*
	 * 		****2-D aggregation****
	 */
	// call 2d reduction to computing mean & std for scatterplot:
	dim3 block_2d( BLOCKSIZE*BLOCKSIZE,1,1);
	dim3 grid_2d ( Nmaps,1,1);
	unsigned int shmem = BLOCKSIZE*BLOCKSIZE*(sizeof(double)); // +sizeof(unsigned char)
	start_t = clock();
	/*
	 * ******Standard Deviation******
	 * DEFINITION:
	 * 	The square root of the average of the squared differences of the values
	 * 	from their average value.
	 * EQUATION:
	 *	sqrt( sum( (A-mean(A)).^2 ) / (numel(A)-1) )
	 * ALGORITHM:
	 * 	1.\ "mean(A)" 						--> reduction_2d_mean
	 * 	2.\ "sum( (A-mean(A)).^2 )"			--> reduction_2d_ssd
	 * 	3.\ "sqrt( ... / (numel(A)-1) )"	--> in "C" (because reduce_2d_std doesn't work!!)
	 * 	******Standard Deviation******
	 */
	reduction_2d_mean<<<grid_2d,block_2d,shmem>>>( dev_iGRIDi, dev_ROI, map_len, Nmaps, dev_oPLOT );
	//reduction_2d_std <<<grid_2d,block_2d,shmem>>>( dev_iGRIDi, map_len, Nmaps, dev_oPLOT );
	reduction_2d_ssd <<<grid_2d,block_2d,shmem>>>( dev_iGRIDi, dev_ROI, map_len, Nmaps, dev_oPLOT );
	CUDA_CHECK_RETURN( cudaMemcpy(oPLOT, dev_oPLOT, oPLOT_bytes, cudaMemcpyDeviceToHost) );
	for(ii=0;ii<Nmaps;ii++){
		oPLOT[Nmaps*2+ii] = oPLOT[ii] + sqrt( oPLOT[Nmaps*1+ii] / (double)(map_len-1) );
		oPLOT[Nmaps*1+ii] = oPLOT[ii] - sqrt( oPLOT[Nmaps*1+ii] / (double)(map_len-1) );
	}
	end_t = clock();

	printf("Implement confidence interval, instead:\n\t[%s]\n\n","https://en.wikipedia.org/wiki/Confidence_interval");

	// save on HDD
	geotiffwrite( iFIL1, oFIL_CUDA, MD, oGRID );

	// save file:
	FILE *fid = fopen(oPLOTcu,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",oPLOTcu); exit(1); }
	for(int rr=0;rr<Nmaps;rr++){ for(int cc=0;cc<Ncols;cc++){ fprintf(fid,"%9.5f ",oPLOT[rr+cc*Nmaps]); } fprintf(fid,"\n"); }
	fclose(fid);

	// print elapsed time:
	printf("%5d [ms]\n\n",(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ) );

	// CUDA free:
	cudaFree( dev_iGRIDi );
	cudaFree( dev_oGRID  );
	cudaFree( dev_oPLOT  );

	return elapsed_time;
}
void whole_mat_single_GPU(){

	metadata 		MD;
	unsigned int	map_len;
	char			DAY[8];
	unsigned int	ii=0;
	int 			elapsed_time_C;
	int 			elapsed_time_CUDA;

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
	MD 					= geotiffinfo( iFIL1, 1 );
	map_len 			= MD.width*MD.heigth;
	// prepare data:
	double	iMap_bytes	= map_len*sizeof( double );
	double	oPLOT_bytes	= Nmaps*3*sizeof( double );
	double *iGRIDi		= (double *) CPLMalloc( iMap_bytes*Nmaps 	);
	unsigned char *roiMap = (unsigned char*) CPLMalloc( map_len*sizeof( unsigned char ) );
	double *oGRID_c		= (double *) CPLMalloc( iMap_bytes 			);
	double *oGRID_cu	= (double *) CPLMalloc( iMap_bytes			);
	double *oPLOT_cu	= (double *) CPLMalloc( oPLOT_bytes 		);
	double *oPLOT_c		= (double *) CPLMalloc( oPLOT_bytes 		);
	double mapDIFF		= 0;
	double plotDIFF		= 0;
	unsigned int loc	= 0;

	// import ROI
	// -roi has different data type:
	metadata MDroi = MD;
	MDroi.pixel_type = GDT_Byte;
	geotiffread( iFIL_ROI, MDroi, &roiMap[0] );

	// import Map using GRID-filename:
	for(ii=0;ii<Nmaps;ii++){
		if(ii<9){ snprintf(DAY,sizeof DAY,"0%d",ii+1); }
		else{snprintf(DAY,sizeof DAY,"%d",ii+1);}
		//printf("DAY:\t%s\n",DAY);
		snprintf(iFILi, sizeof iFILi, "%s%s%s%s%s%s%s%s%s%s%s", iDIR, clPAR, "-", YEAR, MONTH, DAY, "-", SPATMOD, "-", RES, EXT);
		geotiffread( iFILi, MD, &iGRIDi[0] + ii*map_len );
	}

	// *** C (3D) ***
	switch(STAT){
	case 0:
		/*	SUM	*/
		printf("**********\n* %s *\n* %s \n**********\n\n", "-SUM-","single");
		elapsed_time_C 		= C_sum_whole_mat( oGRID_c, iGRIDi, roiMap, map_len, iFIL1, oFIL_C, MD);
		break;

	case 1:
		/*	MIN	*/
		printf("**********\n* %s *\n* %s *\n**********\n\n", "-MIN-","single");
		elapsed_time_C 		= C_min_whole_mat( oGRID_c, iGRIDi, roiMap, map_len, iFIL1, oFIL_C, MD);
		break;

	case 2:
		/*	MAX	*/
		printf("**********\n* %s *\n* %s *\n**********\n\n", "-MAX-","single");
		elapsed_time_C 		= C_max_whole_mat( oGRID_c, iGRIDi, roiMap, map_len, iFIL1, oFIL_C, MD);
		break;

	case 3:
		/*	MEAN*/
		printf("**********\n* %s *\n* %s *\n**********\n\n", "-MEAN-","single");
		elapsed_time_C 		= C_mean_whole_mat( oGRID_c, iGRIDi, roiMap, map_len, iFIL1, oFIL_C, MD);
		break;

	case 4:
		/*	STD	*/
		printf("**********\n* %s *\n* %s *\n**********\n\n", "-STD-","single");
		elapsed_time_C 		= C_std_whole_mat( oGRID_c, iGRIDi, roiMap, map_len, iFIL1, oFIL_C, MD);
		break;
	}
	// *** C (2D) ***
	elapsed_time_C 			= C_std_2D_whole_mat( oPLOT_c, iGRIDi, roiMap, map_len, Nmaps ) + elapsed_time_C;

	// *** CUDA (2D/3D) ***
	elapsed_time_CUDA 		= CUDA_whole_mat(oGRID_cu, oPLOT_cu, iGRIDi, roiMap, map_len, iFIL1, oFIL_CUDA, MD);

	// DIFF
	for( loc=0; loc<map_len; loc++ ) mapDIFF	+= (oGRID_c[loc] - oGRID_cu[loc]);
	for( ii=0; ii<Nmaps; ii++ ) 	 plotDIFF	+= (oPLOT_c[ii]  - oPLOT_cu[ii]);


	// print
	printf( "%12s %5.2f\t[MB]\n",	"iGRIDi size:",	iMap_bytes*Nmaps/1000000			);
	printf( "%12s %5.2f\t[MB]\n\n",	"oGRID size:",	iMap_bytes/1000000					);
	printf( "%12s %5.2f\t[°C]\t%s\n", "mapDIFF:",mapDIFF,"-(C-CUDA)-" 					);
	printf( "%12s %5.2f\t[°C]\t%s\n", "plotDIFF:",plotDIFF,"-(C-CUDA)-" 				);
	printf( "%12s %5d\t[msec]\t%s\n", "Total time:",elapsed_time_C,"-C code-"			);
	printf( "%12s %5d\t[msec]\t%s\n", "Total time:",elapsed_time_CUDA,"-CUDA code-" 	);
	printf( "%12s %5d\t[times]\t%s\n", "Speedup:", (int)((double)(elapsed_time_C)/(double)(elapsed_time_CUDA)),"-C/CUDA-" );

	// GDAL FREE
	VSIFree( iGRIDi		);
	VSIFree( oGRID_c	);
	VSIFree( oGRID_cu	);
}
void chunked_mats_multi_GPU(){

}

int main(int argc, char **argv){


	/*
	 * 		ESTABILISH CONTEXT
	 */
	GDALAllRegister();	// Establish GDAL context.
	cudaFree(0); 		// Establish CUDA context.

	switch( multiGPU ){
		case 0: // SMALL
			whole_mat_single_GPU();
			break;
		case 1: // LARGE
			chunked_mats_multi_GPU();
			break;
	}

	printf("\nFinished!!\n");

	// Destroy context
	CUDA_CHECK_RETURN( cudaDeviceReset() );

	return 0;

	/*
	 * 		..:: CUTTED CODE ::..
	 *
			switch(MD.pixel_type){
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
}
