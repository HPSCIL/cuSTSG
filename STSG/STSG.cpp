//******************************************************************
//STSG is used to reconstruct high - quality NDVI time series data(MODIS / SPOT)
//
//This procedure STSG_v1 is the source code for the first version of STSG.
//Coded by Yang Xue
//
//Reference: Cao, Ruyin, Yang Chen, Miaogen Shen, Jin Chen, Ji Zhou, Cong Wang, and Wei Yang.
//¡°A Simple Method to Improve the Quality of NDVI Time - Series Data by Integrating Spatiotemporal
//Information with the Savitzky - Golay Filter.¡± Remote Sensing of Environment 217 (November 2018) : 244¨C57.
//https ://doi.org/10.1016/j.rse.2018.08.022.
//******************************************************************

#include "gdal_priv.h"

#include "Filter.h"

#include <iostream>
#include <fstream>
using namespace std;

//Input parameters
//******************************************************************
const int year[] = { 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 };

//the thereshold of correlation coefficient to define similar pixels
const float sampcorr = 0.9;

//half of the neighboring window size within which to search similar pixels
const int win = 10;

//the path of the NDVI data
string NDVI_filepath = "../Data/NDVI/NDVI_test_";

//the path of the NDVI quality flags(realibility)
string reliability_filepath = "../Data/Reliability/Reliability_test_";

//snow_address indicates whether to deal with snow contamianted NDVI values(1 = yes / 0 = no)
const int snow_address = 1;
//******************************************************************

int main()
{
	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF");
	char **ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED");

	string filepath = "../Data/";
	int n_elements_year = sizeof(year) / sizeof(year[0]);
	vector<GDALDataset*> fid_NDVI(n_elements_year);
	vector<GDALDataset*> fid_QA(n_elements_year);
	int ns, nl, nb;
	GDALDataType data_type_NDVI, data_type_QA;
	for (int yeari = 0; yeari < n_elements_year; yeari++)
	{
		string filename = NDVI_filepath + to_string(year[yeari]);
		fid_NDVI[yeari] = (GDALDataset*)GDALOpen(filename.c_str(), GA_ReadOnly);
		if (yeari == 0)
		{
			ns = fid_NDVI[yeari]->GetRasterXSize();
			nl = fid_NDVI[yeari]->GetRasterYSize();
			nb = fid_NDVI[yeari]->GetRasterCount();
			data_type_NDVI = fid_NDVI[yeari]->GetRasterBand(1)->GetRasterDataType();
		}

		filename = reliability_filepath + to_string(year[yeari]);
		fid_QA[yeari] = (GDALDataset*)GDALOpen(filename.c_str(), GA_ReadOnly);
		if (yeari == 0)
			data_type_QA = fid_QA[yeari]->GetRasterBand(1)->GetRasterDataType();
	}

	short *imgNDVI = new short[ns*nl*nb];
	unsigned char *imgQA = new unsigned char[ns*nl*nb];
	vector<GDALDataset*> fid_NDVI_buffer(n_elements_year);
	vector<GDALDataset*> fid_QA_buffer(n_elements_year);
	for (int yeari = 0; yeari < n_elements_year; yeari++)
	{
		fid_NDVI[yeari]->RasterIO(GF_Read, 0, 0, ns, nl, imgNDVI, ns, nl, data_type_NDVI, nb, nullptr, 0, 0, 0);
		fid_QA[yeari]->RasterIO(GF_Read, 0, 0, ns, nl, imgQA, ns, nl, data_type_QA, nb, nullptr, 0, 0, 0);

		string outfile = filepath + "NDVI_buffer_" + to_string(year[yeari]) + ".tif";
		fid_NDVI_buffer[yeari] = pDriver->Create(outfile.c_str(), ns + 2 * win, nl + 2 * win, nb, data_type_NDVI, ppszOptions);
		fid_NDVI_buffer[yeari]->RasterIO(GF_Write, win, win, ns, nl, imgNDVI, ns, nl, data_type_NDVI, nb, 0, 0, 0, 0);
		outfile = filepath + "QA_buffer_" + to_string(year[yeari]) + ".tif";
		fid_QA_buffer[yeari] = pDriver->Create(outfile.c_str(), ns + 2 * win, nl + 2 * win, nb, data_type_QA, ppszOptions);
		fid_QA_buffer[yeari]->RasterIO(GF_Write, win, win, ns, nl, imgQA, ns, nl, data_type_QA, nb, 0, 0, 0, 0);

		GDALClose(fid_NDVI[yeari]);
		GDALClose(fid_QA[yeari]);
	}
	delete[] imgNDVI;
	delete[] imgQA;

	ns = ns + 2 * win;
	nl = nl + 2 * win;
	imgNDVI = new short[ns*nl*nb*n_elements_year];
	imgQA = new unsigned char[ns*nl*nb*n_elements_year];
	short *NDVI_reference = new short[ns*nl*nb];
	cout << "Start: Generating NDVI reference time-series data" << endl;
	for (int yeari = 0; yeari < n_elements_year; yeari++)
	{
		fid_NDVI_buffer[yeari]->RasterIO(GF_Read, 0, 0, ns, nl, &imgNDVI[yeari*ns*nl*nb], ns, nl, data_type_NDVI, nb, nullptr, 0, 0, 0);
		fid_QA_buffer[yeari]->RasterIO(GF_Read, 0, 0, ns, nl, &imgQA[yeari*ns*nl*nb], ns, nl, data_type_QA, nb, nullptr, 0, 0, 0);

		GDALClose(fid_NDVI_buffer[yeari]);
		GDALClose(fid_QA_buffer[yeari]);
	}

	for (int i = 0; i < ns; i++)
	{
		for (int j = 0; j < nl; j++)
		{
			int count_vec_res1 = 0;
			int *res_vec_res1 = new int[nb];
			for (int k = 0; k < nb; k++)
			{
				int count_imgQA = 0;
				double mean_imgQA = 0;
				for (int y = 0; y < n_elements_year; y++)
				{
					if (imgQA[i + j*ns + k*ns*nl + y*ns*nl*nb] == 1 || imgQA[i + j*ns + k*ns*nl + y*ns*nl*nb] == 0)
					{
						count_imgQA++;
						mean_imgQA += imgNDVI[i + j*ns + k*ns*nl + y * ns*nl*nb];
					}
				}
				if (count_imgQA >= 3)
				{
					NDVI_reference[i + j*ns + k*ns*nl] = mean_imgQA / count_imgQA;
					res_vec_res1[count_vec_res1++] = k;
				}
				else
					NDVI_reference[i + j*ns + k*ns*nl] = -1.0;
			}

			if (count_vec_res1 < nb && count_vec_res1 >1)
			{
				double k;
				int x = 0;
				int l = 0;
				if (res_vec_res1[count_vec_res1 - 1] != nb - 1)
				{
					k = (NDVI_reference[i + j*ns + (res_vec_res1[count_vec_res1 - 1])*ns*nl] - NDVI_reference[i + j*ns + (res_vec_res1[count_vec_res1 - 2])*ns*nl]) / (res_vec_res1[count_vec_res1 - 1] - res_vec_res1[count_vec_res1 - 2]);
					NDVI_reference[i + j*ns + (nb - 1)*ns*nl] = NDVI_reference[i + j*ns + (res_vec_res1[count_vec_res1 - 1])*ns*nl] + k*(nb - 1 - res_vec_res1[count_vec_res1 - 1]);
					res_vec_res1[count_vec_res1++] = nb - 1;
				}
				int count_res_vec_res1 = count_vec_res1;
				while (count_vec_res1 < nb&&x < count_res_vec_res1 - 1)
				{
					l = res_vec_res1[x + 1] - res_vec_res1[x];
					int n = 1;
					while (l > 1)
					{
						k = (NDVI_reference[i + j*ns + (res_vec_res1[x + 1])*ns*nl] - NDVI_reference[i + j*ns + (res_vec_res1[x])*ns*nl]) / (res_vec_res1[x + 1] - res_vec_res1[x]);
						NDVI_reference[i + j*ns + (res_vec_res1[x] + n)*ns*nl] = NDVI_reference[i + j*ns + (res_vec_res1[x])*ns*nl] + k*n;
						count_vec_res1++;
						n++;
						l--;
					}
					x++;
				}
				if (res_vec_res1[0] != 0)
				{
					k = (NDVI_reference[i + j*ns + (res_vec_res1[1])*ns*nl] - NDVI_reference[i + j*ns + (res_vec_res1[0])*ns*nl]) / (res_vec_res1[1] - res_vec_res1[0]);
					l = res_vec_res1[0];
					int n = 0;
					do
					{
						NDVI_reference[i + j*ns + n*ns*nl] = NDVI_reference[i + j*ns + (res_vec_res1[0])*ns*nl] - k*l;
						n++;
						l--;
					} while (l >= 1);
				}
			}
			delete[] res_vec_res1;
		}
	}

	string outfile = filepath + "NDVI_reference.tif";
	GDALDataset *fid_NDVI_reference = pDriver->Create(outfile.c_str(), ns, nl, nb, data_type_NDVI, ppszOptions);
	fid_NDVI_reference->RasterIO(GF_Write, 0, 0, ns, nl, NDVI_reference, ns, nl, data_type_NDVI, nb, 0, 0, 0, 0);

	cout << "Start: STSG" << endl;
	float *img_NDVI = new float[ns*nl*nb*n_elements_year];
	float *img_QA = new float[ns*nl*nb*n_elements_year];
	float *reference_data = new float[ns*nl*nb];

	for (int i = 0; i < ns; i++)
	{
		for (int j = 0; j < nl; j++)
		{
			for (int k = 0; k < nb; k++)
			{
				for (int y = 0; y < n_elements_year; y++)
				{
					img_NDVI[i + j*ns + k*ns*nl + y*ns*nl*nb] = float(imgNDVI[i + j*ns + k*ns*nl + y*ns*nl*nb]) / 10000;
					img_QA[i + j*ns + k*ns*nl + y*ns*nl*nb] = float(imgQA[i + j*ns + k*ns*nl + y*ns*nl*nb]);
					if (y == 0)
						reference_data[i + j*ns + k*ns*nl] = float(NDVI_reference[i + j*ns + k*ns*nl]) / 10000;
				}
			}
		}
	}
	delete[] imgNDVI;
	delete[] imgQA;
	delete[] NDVI_reference;

	float *vector_out = new float[(ns - 2 * win)*(nl - 2 * win)*nb*n_elements_year];
	STSG_filter(win, sampcorr, img_NDVI, img_QA, reference_data, ns, nl, nb, n_elements_year, snow_address, vector_out);

	delete[] img_NDVI;
	delete[] img_QA;
	delete[] reference_data;

	outfile = filepath + "NDVI_test_STSG.tif";
	GDALDataset *fid_NDVI_test_STSG = pDriver->Create(outfile.c_str(), ns - 2 * win, nl - 2 * win, nb*n_elements_year, GDT_Float32, ppszOptions);
	fid_NDVI_test_STSG->RasterIO(GF_Write, 0, 0, ns - 2 * win, nl - 2 * win, vector_out, ns - 2 * win, nl - 2 * win, GDT_Float32, nb*n_elements_year, 0, 0, 0, 0);

	delete[] vector_out;

	return 0;
}
