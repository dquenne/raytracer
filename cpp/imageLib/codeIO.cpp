// code_io.cpp
//
// read and write our simple .code code file format

// ".code" file format used for storing an arbitrary ensemble of structured light codes
//
// Stores n-band float image, where each band stores a different encoding (Grey, min-SW, XOR-2, or XOR-4)
// Floats are stored in little-endian order.
//
//
//  bytes  contents
//
//  0-3     tag: "CODE" in ASCII, which in little endian happens to be the float 202021.25
//          (just a sanity check that floats are represented correctly)
//  4-7     width as an integer
//  8-11    height as an integer
//  12-15   number of bands/channels in the image
//  16-end  data (width*height*2*4 bytes total)
//          the float values for u and v, interleaved, in row order, i.e.,
//          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
//


// first four bytes, should be the same in little endian
#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "CODE"    // use this when WRITING the file


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "imageLib.h"
#include "codeIO.h"


// read a Code file into 2-band image
void ReadCodeFile(CFloatImage& img, const char* filename)
{
    if (filename == NULL)
	throw CError("ReadCodeFile: empty filename");

    const char *dot = strrchr(filename, '.');
    if (strcmp(dot, ".code") != 0)
	throw CError("ReadCodeFile (%s): extension .code expected", filename);

    FILE *stream = fopen(filename, "rb");
    if (stream == 0)
        throw CError("ReadCodeFile: could not open %s", filename);
    
    int width, height, nBands;
    float tag;

    if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
	(int)fread(&width,  sizeof(int),   1, stream) != 1 ||
	(int)fread(&height, sizeof(int),   1, stream) != 1 ||
	(int)fread(&nBands, sizeof(int),   1, stream) != 1)
	throw CError("ReadCodeFile: problem reading file %s", filename);

    printf("%f",tag);

//    if (tag != TAG_FLOAT) // simple test for correct endian-ness
//	throw CError("ReadCodeFile(%s): wrong tag (possibly due to big-endian machine?)", filename);

    // another sanity check to see that integers were read correctly (99999 should do the trick...)
    if (width < 1 || width > 99999)
	throw CError("ReadCodeFile(%s): illegal width %d", filename, width);

    if (height < 1 || height > 99999)
	throw CError("ReadCodeFile(%s): illegal height %d", filename, height);

    if (nBands < 1 || nBands > 99999)
    throw CError("ReadCodeFile(%s): illegal number of bands %d", filename, nBands);

    CShape sh(width, height, nBands);
    img.ReAllocate(sh);

    //printf("reading %d x %d x 2 = %d floats\n", width, height, width*height*2);
    int n = nBands * width;
    for (int y = 0; y < height; y++) {
	float* ptr = &img.Pixel(0, y, 0);
	if ((int)fread(ptr, sizeof(float), n, stream) != n)
	    throw CError("ReadCodeFile(%s): file is too short", filename);
    }

    if (fgetc(stream) != EOF)
	throw CError("ReadCodeFile(%s): file is too long", filename);

    fclose(stream);
}

// write a 2-band image into Code file
void WriteCodeFile(CFloatImage img, const char* filename)
{
    if (filename == NULL)
	throw CError("WriteCodeFile: empty filename");

    const char *dot = strrchr(filename, '.');
    if (dot == NULL)
	throw CError("WriteCodeFile: extension required in filename '%s'", filename);

    if (strcmp(dot, ".code") != 0)
	throw CError("WriteCodeFile: filename '%s' should have extension '.code'", filename);

    CShape sh = img.Shape();
    int width = sh.width, height = sh.height, nBands = sh.nBands;


    FILE *stream = fopen(filename, "wb");
    if (stream == 0)
        throw CError("WriteCodeFile: could not open %s", filename);

    // write the header
    fprintf(stream, TAG_STRING);
    if ((int)fwrite(&width,  sizeof(int),   1, stream) != 1 ||
	(int)fwrite(&height, sizeof(int),   1, stream) != 1 ||
	(int)fwrite(&nBands, sizeof(int),   1, stream) != 1)
	throw CError("WriteCodeFile(%s): problem writing header", filename);

    // write the rows
    int n = nBands * width;
    for (int y = 0; y < height; y++) {
	float* ptr = &img.Pixel(0, y, 0);
	if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
	    throw CError("WriteCodeFile(%s): problem writing data", filename);
   }

    fclose(stream);
}



