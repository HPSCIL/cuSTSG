//******************************************************************
//cuSTSG is used to reconstruct high-quality NDVI time series data(MODIS/SPOT) based on STSG
//
//This procedure cuSTSG is the source code for the first version of cuSTSG.
//This is a parallel computing code using GPU.
//
//Coded by Yang Xue
//******************************************************************

#include "Filter.h"

#include "gdal_priv.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <iostream>
#include <algorithm>
#include <fstream>

using namespace std;

int main(int argc, char *argv[])
{
	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF");
	char **ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED");

	//parameters
	if (argc != 2)
	{
		cout << "No parameter file!" << endl;
		return 1;
	}

	ifstream parameter(argv[1]);
	if (!parameter)
	{
		cout << "Can't open parameter file!" << endl;
		return 1;
	}

	int* Years = nullptr;
	string NDVI_path, Reliability_path, STSG_Test_path;
	float cosyear, sampcorr;
	int win_year, win, snow_address, n_Years;
	string par;
	while (getline(parameter, par))
	{
		if (par.substr(0, 2) == "//" || par == "")
			continue;

		for (int i = 0; i < par.size(); )
		{
			if (isspace(par[i]))
				par.erase(i,1);
			else
				i++;
		}
		if (par.substr(0, par.find("=")) == "Years")
		{
			vector<int> year;
			while (par.rfind(",") < par.size())
			{
				year.push_back(stoi(par.substr(par.rfind(",") + 1)));
				par = par.substr(0, par.rfind(","));
			}
			year.push_back(stoi(par.substr(par.rfind("=") + 1)));

			n_Years = year.size();
			Years = new int[n_Years];
			for (int i = 0; i < n_Years; i++)
				Years[i] = year[n_Years - i - 1];
		}
		else if (par.substr(0, par.find("=")) == "NDVI_path")
			NDVI_path = par.substr(par.find("=") + 1);
		else if (par.substr(0, par.find("=")) == "Reliability_path")
			Reliability_path = par.substr(par.find("=") + 1);
		else if (par.substr(0, par.find("=")) == "STSG_Test_path")
			STSG_Test_path = par.substr(par.find("=") + 1);
		else if (par.substr(0, par.find("=")) == "cosyear")
			cosyear = stof(par.substr(par.find("=") + 1));
		else if (par.substr(0, par.find("=")) == "win_year")
			win_year = stoi(par.substr(par.find("=") + 1));
		else if (par.substr(0, par.find("=")) == "win")
			win = stoi(par.substr(par.find("=") + 1));
		else if (par.substr(0, par.find("=")) == "sampcorr")
			sampcorr = stof(par.substr(par.find("=") + 1));
		else if (par.substr(0, par.find("=")) == "snow_address")
			snow_address = stoi(par.substr(par.find("=") + 1));
	}
	parameter.close();

	int n_Device;
	cudaGetDeviceCount(&n_Device);
	cudaDeviceProp prop;
	for (int i = 0; i < n_Device; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		cout << "Device " << i << "'s Property:" << endl;
		cout << "Name:" << prop.name << endl;
		cout << "TotalGlobalMem:" << prop.totalGlobalMem / 1024 / 1024 << " MB" << endl;
		cout << "SharedMemPerBlock:" << prop.sharedMemPerBlock / 1024 << " KB" << endl;
		cout << "MaxThreadsPerBlock:" << prop.maxThreadsPerBlock << endl;
		cout << "MaxGridSize:" << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << endl;
		cout << "WarpSize:" << prop.warpSize << endl;
	}

	//cuSTSG
	vector<GDALDataset*> NDVI(n_Years);
	vector<GDALDataset*> QA(n_Years);
	int n_X, n_Y, n_B;
	GDALDataType type_NDVI, type_QA;
	for (int i = 0; i < n_Years; i++)
	{
		string FileName = NDVI_path + to_string(Years[i]);
		NDVI[i] = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
		if (i == 0)
		{
			n_X = NDVI[i]->GetRasterXSize();
			n_Y = NDVI[i]->GetRasterYSize();
			n_B = NDVI[i]->GetRasterCount();
			type_NDVI = NDVI[i]->GetRasterBand(1)->GetRasterDataType();
		}

		FileName = Reliability_path + to_string(Years[i]);
		QA[i] = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
		if (i == 0)
			type_QA = QA[i]->GetRasterBand(1)->GetRasterDataType();
	}

	cout << "Start: cuSTSG" << endl;
	size_t PerYSize = n_X*n_B *(n_Years * sizeof(short) + n_Years * sizeof(unsigned char) + 2 * n_Years * sizeof(float) + sizeof(int) + sizeof(float) + n_Years * sizeof(float)) + n_X*(2 * win + 1)*(2 * win + 1) *(7 * sizeof(float) + 3 * sizeof(int));
	if (prop.totalGlobalMem <= 2 * win*n_X*n_B* (n_Years * sizeof(short) + n_Years * sizeof(unsigned char) + 2 * n_Years * sizeof(float) + sizeof(float)) + n_X*n_Y*n_B*n_Years*sizeof(float))
	{
		cout << "Size of vector_out is larger than totalGlobalMem!" << endl;
		return 1;
	}

	size_t PerStep = (prop.totalGlobalMem - 2 * win*n_X*n_B* (n_Years * sizeof(short) + n_Years * sizeof(unsigned char) + 2 * n_Years * sizeof(float) + sizeof(float)) - n_X*n_Y*n_B*n_Years*sizeof(float)) / PerYSize;
	int Loops = 1;
	if (PerStep < n_Y)
	{
		Loops = n_Y / PerStep + 1;
		PerStep = n_Y / Loops + 1;
	}

	float *d_vector_out;
	size_t nBytes = n_X*n_Y*n_B*n_Years * sizeof(float);
	cudaMalloc((void**)&d_vector_out, nBytes);
	cudaMemset((void*)d_vector_out, 0, nBytes);
	nBytes = win*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
	float *res = (float*)malloc(nBytes);  //(res_cosyear; Slope_res; Intercept_res; new_corr_similar_res;)
	memset((void*)res, 0, nBytes);
	int last_Buffer_Dn = 0;
	for (int i = 1, StartY = 0; i <= Loops&&StartY < n_Y; i++, StartY += PerStep)
	{
		cout << "Loops " << i << endl;
		if (i == Loops)
			PerStep = n_Y - StartY;

		int Buffer_Up = 0;
		int Buffer_Dn = 0;
		if (StartY + PerStep < n_Y - win)
			Buffer_Dn = win;
		else
			Buffer_Dn = n_Y - PerStep - StartY;
		if (StartY >= win)
			Buffer_Up = win;
		else
			Buffer_Up = StartY;

		int blkwidth = 16;
		int blkheight = 16;
		dim3 blocks(blkwidth, blkheight);
		dim3 grids(n_X % blkwidth == 0 ? n_X / blkwidth : n_X / blkwidth + 1, (PerStep + Buffer_Up + Buffer_Dn) % blkheight == 0 ? (PerStep + Buffer_Up + Buffer_Dn) / blkheight : (PerStep + Buffer_Up + Buffer_Dn) / blkheight + 1);

		short *img_NDVI = new short[(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years];
		unsigned char *img_QA = new unsigned char[(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years];
		for (int i = 0; i < n_Years; i++)
		{
			NDVI[i]->RasterIO(GF_Read, 0, StartY - Buffer_Up, n_X, (PerStep + Buffer_Up + Buffer_Dn), &img_NDVI[i*(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B], n_X, (PerStep + Buffer_Up + Buffer_Dn), type_NDVI, n_B, nullptr, 0, 0, 0);
			QA[i]->RasterIO(GF_Read, 0, StartY - Buffer_Up, n_X, (PerStep + Buffer_Up + Buffer_Dn), &img_QA[i*(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B], n_X, (PerStep + Buffer_Up + Buffer_Dn), type_QA, n_B, nullptr, 0, 0, 0);
		}

		short *d_imgNDVI;
		nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years * sizeof(short);
		cudaMalloc((void**)&d_imgNDVI, nBytes);
		cudaMemcpy((void*)d_imgNDVI, (void*)img_NDVI, nBytes, cudaMemcpyHostToDevice);
		unsigned char *d_imgQA;
		nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years * sizeof(unsigned char);
		cudaMalloc((void**)&d_imgQA, nBytes);
		cudaMemcpy((void*)d_imgQA, (void*)img_QA, nBytes, cudaMemcpyHostToDevice);
		float *d_img_NDVI, *d_img_QA;
		nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years *sizeof(float);
		cudaMalloc((void**)&d_img_NDVI, nBytes);
		cudaMalloc((void**)&d_img_QA, nBytes);
		cudaMemset((void*)d_img_NDVI, 0, nBytes);
		cudaMemset((void*)d_img_QA, 0, nBytes);
		float *d_NDVI_Reference, *d_res;
		nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B *  sizeof(float);
		cudaMalloc((void**)&d_NDVI_Reference, nBytes);
		cudaMemset((void*)d_NDVI_Reference, 0, nBytes);
		nBytes = (PerStep + Buffer_Dn)*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
		cudaMalloc((void**)&d_res, nBytes);
		cudaMemset((void*)d_res, 0, nBytes);
		nBytes = last_Buffer_Dn*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
		cudaMemcpy((void*)d_res, (void*)res, nBytes, cudaMemcpyHostToDevice);

		int *d_res_vec_res1;
		nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B * sizeof(int);
		cudaMalloc((void**)&d_res_vec_res1, nBytes);
		cudaMemset((void*)d_res_vec_res1, 0, nBytes);
		float *d_vector_in, *d_res_3;
		nBytes = PerStep*n_X* n_B * sizeof(float);
		cudaMalloc((void**)&d_vector_in, nBytes);
		cudaMemset((void*)d_vector_in, 0, nBytes);
		nBytes = PerStep*n_X*(2 * win + 1)*(2 * win + 1) * 3 * sizeof(float);
		cudaMalloc((void**)&d_res_3, nBytes);//(slope_intercept(2);corr_similar;)
		cudaMemset((void*)d_res_3, 0, nBytes);
		int *d_index;
		nBytes = PerStep*n_X*(2 * win + 1)*(2 * win + 1) * 3 * sizeof(int);
		cudaMalloc((void**)&d_index, nBytes);//(similar_index(2);new_corr;)
		cudaMemset((void*)d_index, 0, nBytes);
		cudaDeviceSynchronize();

		Short_to_Float << <grids, blocks >> >(d_imgNDVI, d_imgQA, n_X, (PerStep + Buffer_Up + Buffer_Dn), n_B, n_Years, d_img_NDVI, d_img_QA);
		cudaDeviceSynchronize();

		Generate_NDVI_reference << <grids, blocks >> >(cosyear, win_year, d_img_NDVI, d_img_QA, n_X, (PerStep + Buffer_Up + Buffer_Dn), n_B, n_Years, d_NDVI_Reference, d_res_3, d_res_vec_res1);
		cudaDeviceSynchronize();

		nBytes = PerStep*n_X*(2 * win + 1)*(2 * win + 1) * 3 * sizeof(float);
		cudaMemset((void*)d_res_3, 0, nBytes);
		cudaDeviceSynchronize();

		Compute_d_res << <grids, blocks >> >(d_img_NDVI, d_img_QA, d_NDVI_Reference, StartY, n_Y, Buffer_Up, Buffer_Dn, n_X, (PerStep + Buffer_Up + Buffer_Dn), n_B, n_Years, win, d_res);
		cudaDeviceSynchronize();

		STSG_filter << <grids, blocks >> >(d_img_NDVI, d_img_QA, d_NDVI_Reference, StartY, n_Y, Buffer_Up, Buffer_Dn, n_X, PerStep, n_B, n_Years, win, sampcorr, snow_address, d_vector_out, d_vector_in, d_res, d_res_3, d_index);
		cudaDeviceSynchronize();

		nBytes = win*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
		memset((void*)res, 0, nBytes);
		nBytes = Buffer_Dn*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
		cudaMemcpy((void*)res, (void*)&d_res[(PerStep + Buffer_Dn - win)*n_X*(2 * win + 1)*(2 * win + 1) * 4], nBytes, cudaMemcpyDeviceToHost);
		last_Buffer_Dn = Buffer_Dn;
		cudaDeviceSynchronize();

		delete[] img_NDVI;
		delete[] img_QA;
		cudaFree((void*)d_imgNDVI);
		cudaFree((void*)d_imgQA);
		cudaFree((void*)d_img_NDVI);
		cudaFree((void*)d_img_QA);
		cudaFree((void*)d_NDVI_Reference);
		cudaFree((void*)d_res);
		cudaFree((void*)d_res_vec_res1);
		cudaFree((void*)d_vector_in);
		cudaFree((void*)d_res_3);
		cudaFree((void*)d_index);
	}
	free((void*)res);
	cudaDeviceSynchronize();

	float *vector_out = new float[n_X*n_Y*n_B*n_Years];
	nBytes = n_X*n_Y*n_B*n_Years* sizeof(float);
	cudaMemcpy((void*)vector_out, (void*)d_vector_out, nBytes, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree((void*)d_vector_out);
	cudaDeviceSynchronize();

	GDALDataset *File = pDriver->Create(STSG_Test_path.c_str(), n_X, n_Y, n_B*n_Years, GDT_Float32, ppszOptions);
	File->RasterIO(GF_Write, 0, 0, n_X, n_Y, vector_out, n_X, n_Y, GDT_Float32, n_B*n_Years, 0, 0, 0, 0);
	GDALClose(File);
	delete[] vector_out;
	
	return 0;
}