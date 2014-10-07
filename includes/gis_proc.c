// current proj includes
#include "/home/giuliano/git/cuda/weatherprog-cudac/includes/gis.h"

// ************* GLOBAL VARs ************* //
GDALDriverH 		hDriver;

metadata geotiffinfo( const char *FIL, unsigned short int iBandNo )
{
	printf("geotiffinfo...");

	metadata			MD;
	GDALDatasetH		iMap;
	GDALRasterBandH		iBand;
	const char			*pszProjectionRef		= NULL;
	double        		iGeoTransform[6];
	int 				*pbSuccess=0;
	double				nodata_value;
	const char			*pszAuthorityName		= NULL;
	const char			*pszAuthorityCode 		= NULL;
	/*
	 * 	initialization
	 */
	MD.nodata_value		= NULL;								// nodata value (NULL by default)
	MD.flag				= 0;								// flag about nodata  (0 by default) ("has nodata value" = FALSE) and ("isnodata" = FALSE necessarily)
	MD.skewX			= 0;								// rotation parameter (0 by default)
	MD.skewY			= 0;								// rotation parameter (0 by default)

	/*
	 * 	ESTABILISH CONTEXT
	 */
	GDALAllRegister();	// Establish GDAL context.

	/*
	 *	Open MAP:
	 */
	iMap 				= GDALOpen( FIL, GA_ReadOnly );
	if( iMap == NULL ){ printf("Error: cannot load input grids!\n"); exit(EXIT_FAILURE);}

	// driver of iMap's
	hDriver 			= GDALGetDatasetDriver( iMap );

	GDALGetGeoTransform( iMap, iGeoTransform );
	// get BANDS:
	iBand				= GDALGetRasterBand(iMap, iBandNo); // GDALGetRasterCount( iMap1 ) is equal to 1.
	nodata_value		= GDALGetRasterNoDataValue( iBand, pbSuccess );

	/*********** Spatial Reference System **********
	 * http://postgis.refractions.net/documentation/postgis-doxygen/d7/d0a/raster2pgsql_8c-source.html	(search for hSRS --> line 1464-1467)
	 * http://www.gdal.org/ogr/classOGRSpatialReference.html#a178f4593ef4ee661f2df9d221feaa803
	 * 	const char* OSRGetAuthorityCode	(	OGRSpatialReferenceH 	hSRS, const char * 	pszTargetKey );
	 * 		pszTargetKey 	the partial or complete path to the node to get an authority from.
	 * 						ie. "PROJCS", "GEOGCS", "GEOGCS|UNIT" or NULL to search for an
	 * 						authority node on the root element.
	 */
	pszProjectionRef	= GDALGetProjectionRef( iMap );
	if( pszProjectionRef == NULL ){ printf("Error: one or more input layers miss spatial reference system!\n"); exit(EXIT_FAILURE); }
	OGRSpatialReferenceH hSRS = OSRNewSpatialReference(NULL);
	if( OSRSetFromUserInput(hSRS, pszProjectionRef) == OGRERR_NONE ){
		pszAuthorityName = OSRGetAuthorityName(hSRS, NULL);
		pszAuthorityCode = OSRGetAuthorityCode(hSRS, NULL);
	}
	//printf( "Projection is:\t\t\t%s:%s\n", pszAuthorityName, pszAuthorityCode );

	// define METADATA:
	MD.scaleX			= iGeoTransform[1];					// pixel width
	MD.scaleY			= iGeoTransform[5];					// pixel height
	MD.ipX				= iGeoTransform[0];					// upper-left corner of upper-left pixel
	MD.ipY				= iGeoTransform[3];					// upper-left corner of upper-left pixel
	MD.SRID_name		= pszAuthorityName;					// Get the authority name
	MD.SRID				= pszAuthorityCode;					// the spatial reference identifier
	MD.WKTproj			= GDALGetProjectionRef( iMap );		// the projection coordinate system of the image in OpenGIS WKT format
	MD.heigth			= GDALGetRasterYSize( iMap );		// nRows
	MD.width			= GDALGetRasterXSize( iMap );		// nCols
	MD.skewX			= 0;								// rotation parameter (0 by default)
	MD.skewY			= 0;								// rotation parameter (0 by default)
	MD.nBands			= GDALGetRasterCount( iMap );		// number of bands
	if(&pbSuccess){
	MD.nodata_value		= nodata_value;						// nodata value (NULL by default)
	MD.flag				= 4;								// ("has nodata value" = TRUE ) and ("isnodata" = FALSE)
	}
	MD.pixel_type		= GDALGetRasterDataType( iBand );	// raster band data type

	if( MD.scaleX != abs(MD.scaleY) ) { printf("Warning: Pixel is not squared [%f x %f]!\n", MD.scaleX, MD.scaleY); }

	GDALClose( iMap );

	printf("\t\t...END!\n");

	return MD;
	/* GDAL DATA TYPES:
	 * 		GDT_Unknown = 0, GDT_Byte 	= 1, GDT_UInt16   = 2,  GDT_Int16 	 = 3,
	 *		GDT_UInt32 	= 4, GDT_Int32 	= 5, GDT_Float32  = 6,  GDT_Float64  = 7,
	 *		GDT_CInt16 	= 8, GDT_CInt32 = 9, GDT_CFloat32 = 10, GDT_CFloat64 = 11,
	 *		GDT_TypeCount = 12
	 */

}

template <class T>
void geotiffread( const char *FIL, metadata MD, T *host_iMap ){

	printf("geotiffread...");

	GDALDatasetH		iMap;
	GDALRasterBandH		iBand;
	double				iMap_bytes;

	/*
	 *	Open MAP:
	 */
	iMap 				= GDALOpen( FIL, GA_ReadOnly );
	if( iMap == NULL ){ printf("Error: cannot load input grids!\n"); exit(EXIT_FAILURE);}
	hDriver 			= GDALGetDatasetDriver( iMap ); // driver of iMap's

	if(GDALGetRasterCount( iMap )>1 ){
		printf("Error: iMap has more than 1 band! [not allowed]\n");
		exit(EXIT_FAILURE);
	}

	/* GDAL DATA TYPES:
	 * 		GDT_Unknown = 0, GDT_Byte 	= 1, GDT_UInt16   = 2,  GDT_Int16 	 = 3,
	 *		GDT_UInt32 	= 4, GDT_Int32 	= 5, GDT_Float32  = 6,  GDT_Float64  = 7,
	 *		GDT_CInt16 	= 8, GDT_CInt32 = 9, GDT_CFloat32 = 10, GDT_CFloat64 = 11,
	 *		GDT_TypeCount = 12
	 */
	iMap_bytes			= MD.heigth*MD.width*sizeof( T );
	host_iMap			= (float *) CPLMalloc( iMap_bytes );
	iBand				= GDALGetRasterBand( iMap,  GDALGetRasterCount( iMap  ) );
	//GDT_type			= GDALGetRasterDataType	( iBand );
	GDALRasterIO( iBand,  GF_Read, 0, 0, MD.width, MD.heigth, host_iMap,  MD.width, MD.heigth, MD.pixel_type, 0, 0 );

	GDALClose( iMap );

	printf("\t\t...END!\n");
}

template <class T>
void geotiffwrite( const char *refFIL, const char *FIL, metadata MD, T *host_oMap ){

	printf("geotiffwrite...");

	/*
	 * 	DECLARATIONS
	 */
	char 				**papszOptions = NULL;
	GDALDatasetH		oMap,iMap;
	GDALRasterBandH		oBand;
	double        		iGeoTransform[6];

	iMap 				= GDALOpen( refFIL, GA_ReadOnly );
	if( iMap == NULL ){ printf("Error: cannot load REF grid!\n"); exit(EXIT_FAILURE);}
	hDriver 			= GDALGetDatasetDriver( iMap );

	/* adfGeoTransform[0] // top left x
     * adfGeoTransform[1] // w-e pixel resolution
     * adfGeoTransform[2] // 0
     * adfGeoTransform[3] // top left y
     * adfGeoTransform[4] // 0
     * adfGeoTransform[5] // n-s pixel resolution (negative value)
	 */
	iGeoTransform[0]	= MD.ipX;
	iGeoTransform[1]	= MD.scaleX;
	iGeoTransform[2]	= 0;
	iGeoTransform[3]	= MD.ipY;
	iGeoTransform[4]	= 0;
	iGeoTransform[5]	= MD.scaleY;

	// CREATE DATASET on HDD
	//	-options: tiling with block 512x512
	papszOptions = CSLSetNameValue( papszOptions, "TILED", "YES");
	papszOptions = CSLSetNameValue( papszOptions, "BLOCKXSIZE", "512");
	papszOptions = CSLSetNameValue( papszOptions, "BLOCKYSIZE", "512");

	//	-instantiate GRID
	oMap = GDALCreate( hDriver, FIL, MD.width, MD.heigth, 1, MD.pixel_type, papszOptions );// GDT_Float32

	//	-set projection
	GDALSetProjection( 	 oMap, GDALGetProjectionRef( iMap ) );

	//	-set geospatial transformation
	GDALSetGeoTransform( oMap, iGeoTransform );
	//	-band
	oBand = GDALGetRasterBand( oMap, 1 );

	//	-write to HDD
	GDALRasterIO( oBand, GF_Write, 0, 0, MD.width, MD.heigth, host_oMap, MD.width, MD.heigth, MD.pixel_type, 0, 0 );

	GDALClose( oMap );
	CSLDestroy( papszOptions );

	printf("\t\t...END!\n");
}
