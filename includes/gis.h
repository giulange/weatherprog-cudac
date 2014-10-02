#include <math.h>
#include <inttypes.h>
#include <stdio.h>

// GDAL
#include <gdal.h>
#include "cpl_conv.h" 		/* for CPLMalloc() */
#include "cpl_string.h"
#include "ogr_srs_api.h"
//#include "gdal_priv.h"

struct CPtrArray{
	char *m_WorkingDirectory;
	char *m_FileName;
	uint32_t *m_Row;
	uint32_t *m_Column;
	double *m_CellSize;
	double *m_NoDataValue;
	uint32_t *m_PitArea;
	double *m_PitDepth;
	bool m_InvertedElev;
};

struct metadata{
//	TYPE			VAR					DEFINITIONS in PostGIS  (see --> http://postgis.net/docs/RT_reference.html)
	uint16_t 		nBands;			//	the number of bands in the raster object
	double 			scaleX;			//	the X component of the pixel width in units of coordinate reference system
	double 			scaleY;			//	the Y component of the pixel height in units of coordinate reference system
	double 			ipX;			//	the upper left X coordinate of raster in projected spatial ref
	double 			ipY;			//	the upper left Y coordinate of raster in projected spatial ref
	double 			skewX;			//	the georeference X skew (or rotation parameter)
	double 			skewY;			//	the georeference Y skew (or rotation parameter)
	const char		*SRID;			//	the spatial reference identifier of the raster (as defined in spatial_ref_sys table)
	uint32_t 		width;			//	the width of the raster in pixels	(ncols)
	uint32_t 		heigth;			//	the height of the raster in pixels	(nrows)
									//	-----------------------------------
	uint16_t		flag;			//	FLAG with following cases:
									//		0	-->	("has nodata value" = FALSE) and ("isnodata" = FALSE necessarily)
									//		4	--> ("has nodata value" = TRUE ) and ("isnodata" = FALSE			)
									//		6	-->	("has nodata value" = TRUE ) and ("isnodata" = TRUE				)
									//		8	-->	("ext" flag)
									//	> "has nodata" 	-->	a flag indicating if this band contains nodata values.
									//	> "isnodata" 	-->	a flag indicating if this band is filled only with nodata values.
									//						The flag CANNOT be TRUE if hasnodata is FALSE.
									//	-----------------------------------
	GDALDataType	pixel_type;		//	Band data type (with the FLAG explained above):
    								//		00 - 1BB   -  1-bit boolean,
                    				//		01 - 2BUI  -  2-bit unsigned integer,
                    				//		02 - 4BUI  -  4-bit unsigned integer,
                    				//		03 - 8BSI  -  8-bit signed integer,
                    				//		04 - 8BUI  -  8-bit unsigned integer,
                    				//		05 - 16BSI - 16-bit signed integer,
                    				//		06 - 16BUI - 16-bit unsigned integer,
                    				//		07 - 32BSI - 32-bit signed integer,
                    				//		08 - 32BUI - 32-bit unsigned integer,
                    				//		0A - 32BF  - 32-bit float,
                    				//		0B - 64BF  - 64-bit float + hasnodatavalue set to false,
                    				//		4B - 64BF  - 64-bit float + hasnodatavalue set to true and isnodata set to false,
                    				//		6B - 64BF  - 64-bit float + 6 -> hasnodatavalue and isnodata set to true,
                    				//		8B - 64BF  - 64-bit float + ext flag (with two more fields{ext band num == 3,"/tmp/t.tif"}
									//	-----------------------------------
	double			nodata_value;	//	the value in the given band that represents no data
};


metadata geotiffinfo( const char *FIL, unsigned short int iBandNo );

template <class T>
void geotiffread( const char *FIL, metadata MD, T *host_iMap );

template <class T>
uint32_t flow_mapr_proc(T *DEMgrid, CPtrArray InputArr);

