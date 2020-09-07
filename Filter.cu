#include "Filter.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <algorithm>
#include <numeric>
#include <stdio.h>

using namespace std;

__global__ void Short_to_Float(short *imgNDVI, unsigned char *imgQA, int n_X, int n_Y, int n_B, int n_Years, float *img_NDVI, float *img_QA)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_X)
		return;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= n_Y)
		return;

	for (int k = 0; k < n_B; k++)
	{
		for (int y = 0; y < n_Years; y++)
		{
			img_NDVI[i + j*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] = float(imgNDVI[i + j*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B]) / 10000;
			img_QA[i + j*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] = float(imgQA[i + j*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B]);

			if (img_NDVI[i + j*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] <= -1 || img_NDVI[i + j*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] >= 1
				|| img_QA[i + j*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] < -1 || img_QA[i + j*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B]>3)
				img_QA[i + j*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] = -1;
		}
	}
}

__global__ void Generate_NDVI_reference(float cosyear, float *img_NDVI, float *img_QA, int n_X, int n_Y, int n_B, int n_Years, float *reference_data, float *d_res_3, int *d_res_vec_res1)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_X)
		return;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= n_Y)
		return;

	//calculating cosine similarity
	float *res_cosyear = &d_res_3[(i + j*n_X)*(n_Years - 1)];
	for (int y = 0; y < n_Years - 1; y++)
	{
		double xy_sum = 0;
		double x2_sum = 0;
		double y2_sum = 0;
		for (int k = 0; k < n_B; k++)
		{
			if ((img_QA[i + j *n_X + k*n_X *n_Y + y*n_X*n_Y*n_B] == 0 || img_QA[i + j *n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] == 1)
				&& (img_QA[i + j *n_X + k*n_X *n_Y + (y + 1)*n_X*n_Y*n_B] == 0 || img_QA[i + j *n_X + k*n_X*n_Y + (y + 1)*n_X*n_Y*n_B] == 1))
			{
				xy_sum += img_NDVI[i + j *n_X + k*n_X *n_Y + y*n_X*n_Y*n_B] * img_NDVI[i + j *n_X + k*n_X *n_Y + (y + 1)*n_X*n_Y*n_B];
				x2_sum += img_NDVI[i + j *n_X + k*n_X *n_Y + y*n_X*n_Y*n_B] * img_NDVI[i + j *n_X + k*n_X *n_Y + y*n_X*n_Y*n_B];
				y2_sum += img_NDVI[i + j *n_X + k*n_X *n_Y + (y + 1)*n_X*n_Y*n_B] * img_NDVI[i + j *n_X + k*n_X *n_Y + (y + 1)*n_X*n_Y*n_B];
			}
		}
		res_cosyear[y] = xy_sum / sqrt(x2_sum*y2_sum);
	}

	int count_vec_res1 = 0;
	int *res_vec_res1 = &d_res_vec_res1[(i + j*n_X)*n_B];
	for (int k = 0; k < n_B; k++)
	{
		int count_img_QA = 0;
		float mean_img_QA = 0;
		if ((res_cosyear[0] >= cosyear) && (img_QA[i + j *n_X + k*n_X *n_Y] == 0 || img_QA[i + j *n_X + k*n_X*n_Y] == 1))
		{
			count_img_QA++;
			mean_img_QA += img_NDVI[i + j *n_X + k*n_X*n_Y];
		}
		for (int y = 0; y < n_Years - 1; y++)
		{
			if ((res_cosyear[y] >= cosyear) && (img_QA[i + j *n_X + k*n_X *n_Y + (y + 1)*n_X*n_Y*n_B] == 0 || img_QA[i + j *n_X + k*n_X*n_Y + (y + 1)*n_X*n_Y*n_B] == 1))
			{
				count_img_QA++;
				mean_img_QA += img_NDVI[i + j *n_X + k*n_X*n_Y + (y + 1)*n_X*n_Y*n_B];
			}
		}
		if (count_img_QA >= 1)
		{
			reference_data[i + j*n_X + k*n_X*n_Y] = mean_img_QA / count_img_QA;
			res_vec_res1[count_vec_res1++] = k;
		}
	}

	if (count_vec_res1 < n_B && count_vec_res1 > 1)
	{
		double k;
		int x = 0;
		int l = 0;
		if (res_vec_res1[count_vec_res1 - 1] != n_B - 1)
		{
			k = (reference_data[i + j*n_X + res_vec_res1[count_vec_res1 - 1] * n_X*n_Y] - reference_data[i + j*n_X + res_vec_res1[count_vec_res1 - 2] * n_X*n_Y]) / (res_vec_res1[count_vec_res1 - 1] - res_vec_res1[count_vec_res1 - 2]);
			reference_data[i + j*n_X + (n_B - 1)*n_X*n_Y] = reference_data[i + j*n_X + (res_vec_res1[count_vec_res1 - 1])*n_X*n_Y] + k*(n_B - 1 - res_vec_res1[count_vec_res1 - 1]);
			res_vec_res1[count_vec_res1++] = n_B - 1;
		}
		int count_res_vec_res1 = count_vec_res1;
		while (count_vec_res1 < n_B&&x < count_res_vec_res1 - 1)
		{
			l = res_vec_res1[x + 1] - res_vec_res1[x];
			int n = 1;
			while (l > 1)
			{
				k = (reference_data[i + j*n_X + res_vec_res1[x + 1] * n_X*n_Y] - reference_data[i + j*n_X + res_vec_res1[x] * n_X*n_Y]) / (res_vec_res1[x + 1] - res_vec_res1[x]);
				reference_data[i + j*n_X + (res_vec_res1[x] + n)*n_X*n_Y] = reference_data[i + j*n_X + res_vec_res1[x] * n_X*n_Y] + k*n;
				count_vec_res1++;
				n++;
				l--;
			}
			x++;
		}
		if (res_vec_res1[0] != 0)
		{
			k = (reference_data[i + j*n_X + res_vec_res1[1] * n_X*n_Y] - reference_data[i + j*n_X + res_vec_res1[0] * n_X*n_Y]) / (res_vec_res1[1] - res_vec_res1[0]);
			l = res_vec_res1[0];
			int n = 0;
			do
			{
				reference_data[i + j*n_X + n*n_X*n_Y] = reference_data[i + j*n_X + res_vec_res1[0] * n_X*n_Y] - k*l;
				n++;
				l--;
			} while (l >= 1);
		}
	}
}

__global__ void Compute_d_res(float *img_NDVI, float*img_QA, float *reference_data, int StartY, int TotalY, int Buffer_Up, int Buffer_Dn, int n_X, int n_Y, int n_B, int n_Years, int win, float *d_res)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_X)
		return;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= n_Y - Buffer_Dn- Buffer_Up )
		return;

	float *corr_res = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4];
	float *Slope_res = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1)];
	float *Intercept_res = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 2];
	float *new_corr_similar_res = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 3];
	for (int sj = 0; sj <= win; sj++)
	{
		for (int si = -1 * win; si <= win; si++)
		{
			if (i + si >= 0 && i + si < n_X&&j + sj + StartY >= 0 && j + sj + StartY < TotalY)
			{
				double x_sum = 0;
				double y_sum = 0;
				double x_mean = 0;
				double y_mean = 0;
				double xy_sum = 0;
				double x2_sum = 0;
				double y2_sum = 0;
				for (int k = 0; k < n_B; k++)
				{
					x_sum += reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y];
					y_sum += reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y];
				}
				x_mean = x_sum / n_B;
				y_mean = y_sum / n_B;
				for (int k = 0; k < n_B; k++)
				{
					xy_sum += (reference_data[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] - x_mean) * (reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y] - y_mean);
					x2_sum += (reference_data[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] - x_mean) * (reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] - x_mean);
					y2_sum += (reference_data[i + (j + Buffer_Up)*n_X + k *n_X*n_Y] - y_mean) * (reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y] - y_mean);
				}
				corr_res[si + win + (sj + win)*(2 * win + 1)] = xy_sum / sqrt(x2_sum*y2_sum);
				d_res[(i + si + (j + sj)*n_X)*(2 * win + 1)*(2 * win + 1) * 4 - si + win + (-1 * sj + win)*(2 * win + 1)] = xy_sum / sqrt(x2_sum*y2_sum);

				int count_tempQA = 0;
				x_sum = 0;
				y_sum = 0;
				x_mean = 0;
				y_mean = 0;
				xy_sum = 0;
				x2_sum = 0;
				y2_sum = 0;
				for (int k = 0; k < n_B; k++)
				{
					for (int y = 0; y < n_Years; y++)
					{
						if ((img_QA[i + (j + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] == 0 || img_QA[i + (j + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] == 1)
							&& (img_QA[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] == 0 || img_QA[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] == 1)
							&& reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y] != 0 && reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] != 0)
						{
							count_tempQA++;
							x_sum += img_NDVI[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y];
							y_sum += img_NDVI[i + (j + Buffer_Up) *n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y];
							xy_sum += (img_NDVI[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] - x_mean) * (img_NDVI[i + (j + Buffer_Up) *n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y] - y_mean);
							x2_sum += (img_NDVI[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] - x_mean) * (img_NDVI[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] - x_mean);
							y2_sum += (img_NDVI[i + (j + Buffer_Up) *n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y] - y_mean) * (img_NDVI[i + (j + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y] - y_mean);
						}
					}
				}
				if (count_tempQA >= 30)
				{
					Slope_res[si + win + (sj + win)*(2 * win + 1)] = (xy_sum*count_tempQA - x_sum*y_sum) / (x2_sum*count_tempQA - x_sum*x_sum);
					Intercept_res[si + win + (sj + win)*(2 * win + 1)] = (x2_sum*y_sum - x_sum*xy_sum) / (x2_sum*count_tempQA - x_sum*x_sum);
					d_res[(i + si + (j + sj)*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) - si + win + (-1 * sj + win)*(2 * win + 1)] = (xy_sum*count_tempQA - y_sum*x_sum) / (y2_sum*count_tempQA - y_sum*y_sum);
					d_res[(i + si + (j + sj)*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 2 - si + win + (-1 * sj + win)*(2 * win + 1)] = (y2_sum*x_sum - y_sum*xy_sum) / (y2_sum*count_tempQA - y_sum*y_sum);

					x_mean = x_sum / count_tempQA;
					y_mean = y_sum / count_tempQA;
					xy_sum = 0;
					x2_sum = 0;
					y2_sum = 0;
					for (int k = 0; k < n_B; k++)
					{
						for (int y = 0; y < n_Years; y++)
						{
							if ((img_QA[i + (j + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] == 0 || img_QA[i + (j + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] == 1)
								&& (img_QA[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] == 0 || img_QA[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] == 1)
								&& reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y] != 0 && reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] != 0)
							{
								xy_sum += (img_NDVI[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] - x_mean) * (img_NDVI[i + (j + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y] - y_mean);
								x2_sum += (img_NDVI[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] - x_mean) * (img_NDVI[i + si + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + k*n_X*n_Y] - x_mean);
								y2_sum += (img_NDVI[i + (j + Buffer_Up)*n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y] - y_mean) * (img_NDVI[i + (j + Buffer_Up) *n_X + k*n_X*n_Y + y*n_X*n_Y*n_B] / reference_data[i + (j + Buffer_Up)*n_X + k*n_X*n_Y] - y_mean);
							}
						}
					}
					new_corr_similar_res[si + win + (sj + win)*(2 * win + 1)] = xy_sum / sqrt(x2_sum*y2_sum);
					d_res[(i + si + (j + sj)*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 3 - si + win + (-1 * sj + win)*(2 * win + 1)] = xy_sum / sqrt(x2_sum*y2_sum);
				}

				if (reference_data[(i + si) + (j + sj + Buffer_Up)*n_X + 3 * n_X*n_Y] == 0)
				{
					corr_res[si + win + (sj + win)*(2 * win + 1)] = 0;
					Slope_res[si + win + (sj + win)*(2 * win + 1)] = 0;
					Intercept_res[si + win + (sj + win)*(2 * win + 1)] = 0;
					new_corr_similar_res[si + win + (sj + win)*(2 * win + 1)] = 0;
				}
				if (reference_data[i + (j + Buffer_Up)*n_X + 3 * n_X*n_Y] == 0)
				{
					d_res[(i + si + (j + sj)*n_X)*(2 * win + 1)*(2 * win + 1) * 4 - si + win + (-1 * sj + win)*(2 * win + 1)] = 0;
					d_res[(i + si + (j + sj)*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) - si + win + (-1 * sj + win)*(2 * win + 1)] = 0;
					d_res[(i + si + (j + sj)*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 2 - si + win + (-1 * sj + win)*(2 * win + 1)] = 0;
					d_res[(i + si + (j + sj)*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 3 - si + win + (-1 * sj + win)*(2 * win + 1)] = 0;
				}
			}
		}
	}
}

__global__ void STSG_filter(float *img_NDVI, float *img_QA, float *reference_data, int StartY, int TotalY, int Buffer_Up, int Buffer_Dn, int n_X, int n_Y, int n_B, int n_Years, int win, float sampcorr, int snow_address, float *vector_out, float *d_vector_in, float *d_res, float *d_res_3, int *d_index)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_X)
		return;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= n_Y)
		return;

	int samp = 0;
	int aap = 0;
	int *similar_index = &d_index[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 3];
	float *slope_intercept = &d_res_3[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 3];
	float *corr_similar = &d_res_3[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 3 + (2 * win + 1)*(2 * win + 1) * 2];
	for (int y = 0; y < n_Years; y++)
	{
		float *vector_in = &d_vector_in[(i + j*n_X)*n_B];
		for (int k = 0; k < n_B; k++)
			vector_in[k] = img_NDVI[i + (j + Buffer_Up) *n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B];

		float vector_in_max_1 = vector_in[0];
		float vector_in_max_2 = vector_in[1];
		float vector_in_max_3 = vector_in[2];
		if (vector_in_max_2 >= vector_in_max_1)
		{
			vector_in_max_2 = vector_in_max_1;
			vector_in_max_1 = vector_in[1];
		}
		if (vector_in_max_3 >= vector_in_max_1)
		{
			vector_in_max_3 = vector_in_max_2;
			vector_in_max_2 = vector_in_max_1;
			vector_in_max_1 = vector_in[2];
		}
		else if (vector_in_max_3 >= vector_in_max_2&&vector_in_max_3 < vector_in_max_1)
		{
			vector_in_max_3 = vector_in_max_2;
			vector_in_max_2 = vector_in[2];
		}
		for (int k = 3; k < n_B; k++)
		{
			if (vector_in[k] >= vector_in_max_1)
			{
				vector_in_max_3 = vector_in_max_2;
				vector_in_max_2 = vector_in_max_1;
				vector_in_max_1 = vector_in[k];
			}
			else if (vector_in[k] >= vector_in_max_2&&vector_in[k] < vector_in_max_1)
			{
				vector_in_max_3 = vector_in_max_2;
				vector_in_max_2 = vector_in[k];
			}
			else if (vector_in[k] >= vector_in_max_3&&vector_in[k] < vector_in_max_2)
				vector_in_max_3 = vector_in[k];
		}

		if (((vector_in_max_1 + vector_in_max_2 + vector_in_max_3) / 3) > 0.15)
		{
			int indic = 0;
			//searching similar pixels
			if (y == 0)
			{
				float *corr_res = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4];
				float *Slope_res = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1)];
				float *Intercept_res = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 2];
				float *new_corr_similar_res = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 3];
				int count_corr_Slope = 0;
				for (int m = 0; m < (2 * win + 1)*(2 * win + 1); m++)
				{
					if (corr_res[m] >= sampcorr&&Slope_res[m] != 0.)
						count_corr_Slope++;
				}
				if (count_corr_Slope >= 2)
				{
					samp = count_corr_Slope - 1;
					int *new_corr = &d_index[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 3 + (2 * win + 1)*(2 * win + 1) * 2];
					for (int m = 0; m < (2 * win + 1)*(2 * win + 1); m++)
						new_corr[m] = m;
					for (int m = 0; m < count_corr_Slope; m++)
					{
						for (int n = m + 1; n < (2 * win + 1)*(2 * win + 1); n++)
						{
							if (corr_res[new_corr[m]] < corr_res[new_corr[n]])
							{
								int temp = new_corr[m];
								new_corr[m] = new_corr[n];
								new_corr[n] = temp;
							}
						}
					}

					for (int k = 0; k < samp; k++)
					{
						similar_index[1 + k * 2] = int(new_corr[k + 1] / (2 * win + 1)) + j - win;
						similar_index[k * 2] = new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1))*(2 * win + 1) + i - win;
						slope_intercept[1 + k * 2] = Slope_res[int(new_corr[k + 1] / (2 * win + 1)) + (2 * win + 1)*(new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1))*(2 * win + 1))];
						slope_intercept[k * 2] = Intercept_res[int(new_corr[k + 1] / (2 * win + 1)) + (2 * win + 1)*(new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1))*(2 * win + 1))];
						corr_similar[k] = new_corr_similar_res[int(new_corr[k + 1] / (2 * win + 1)) + (2 * win + 1)* (new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1))*(2 * win + 1))];
					}
					aap = 1;
				}
				else
					aap = 0;
			}

			//generating the trend curve
			float *trend_NDVI = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4];
			if (aap == 1)
			{
				int count_trend_NDVI = 0;
				int nocount_trend_NDVI = 0;
				int *res_trend_NDVI = &d_index[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 3 + (2 * win + 1)*(2 * win + 1) * 2];
				int *nores_trend_NDVI = &res_trend_NDVI[n_B];
				int count_conres = 0;
				for (int k = 0; k < n_B; k++)
				{
					float *temp_NDVI = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1)];
					int count_temp_NDVI = 0;
					float total_new_corr_similar = 0;
					float total_new_temp = 0;
					for (int m = 0; m < samp; m++)
					{
						if (similar_index[m * 2] >= 0 && similar_index[m * 2] < n_X && similar_index[m * 2 + 1] + Buffer_Up >= 0 && similar_index[m * 2 + 1] + Buffer_Up < TotalY
							&& (img_QA[similar_index[m * 2] + (similar_index[1 + m * 2] + Buffer_Up) * n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 0 || img_QA[similar_index[m * 2] + (similar_index[1 + m * 2] + Buffer_Up) * n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 1)
							&& reference_data[similar_index[m * 2] + (similar_index[1 + m * 2] + Buffer_Up) *n_X + k*n_X * (n_Y + Buffer_Up + Buffer_Dn)] != 0)
						{
							float new_ratio = img_NDVI[similar_index[m * 2] + (similar_index[1 + m * 2] + Buffer_Up) * n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] / reference_data[similar_index[m * 2] + (similar_index[1 + m * 2] + Buffer_Up)* n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn)];
							temp_NDVI[m] = (slope_intercept[m * 2] + new_ratio*slope_intercept[1 + m * 2])*reference_data[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn)];
							if (temp_NDVI[m] >= 1 || temp_NDVI[m] <= -0.2)
								temp_NDVI[m] = 0;
							if (temp_NDVI[m] != 0)
							{
								count_temp_NDVI++;
								total_new_corr_similar += corr_similar[m];
							}
						}
						else
							temp_NDVI[m] = 0;
					}
					if (count_temp_NDVI != 0)
					{
						for (int m = 0; m < samp; m++)
						{
							if (temp_NDVI[m] != 0)
								total_new_temp += corr_similar[m] / total_new_corr_similar * temp_NDVI[m];
						}
						trend_NDVI[k] = total_new_temp;
						if (trend_NDVI[k] != 0)
						{
							res_trend_NDVI[count_trend_NDVI++] = k;
							if (count_trend_NDVI > 1 && k - res_trend_NDVI[count_trend_NDVI - 2] >= 3)
								count_conres++;
						}
						else
							nores_trend_NDVI[nocount_trend_NDVI++] = k;
					}
					else
					{
						trend_NDVI[k] = 0;
						nores_trend_NDVI[nocount_trend_NDVI++] = k;
					}
				}

				//generating the trend NDVI
				if (count_trend_NDVI >= n_B / 2 && count_conres == 0)
				{
					for (int m = 0; m < nocount_trend_NDVI; m++)
					{
						int sta = 0;
						if (nores_trend_NDVI[m] < res_trend_NDVI[0])
							sta = 0;
						else if (nores_trend_NDVI[m] > res_trend_NDVI[count_trend_NDVI - 1])
							sta = count_trend_NDVI - 4;
						else
						{
							for (int n = 0; n < count_trend_NDVI - 1; n++)
							{
								if (res_trend_NDVI[n] < nores_trend_NDVI[m] && nores_trend_NDVI[m] < res_trend_NDVI[n + 1])
								{
									if (n - 1 < 0)
										sta = 0;
									else if (count_trend_NDVI - n < 4)
										sta = count_trend_NDVI - 4;
									else
										sta = n - 1;
									break;
								}
							}
						}

						float x[4], y[4];
						for (int n = 0; n < 4; n++)
						{
							x[n] = res_trend_NDVI[sta + n];
							y[n] = trend_NDVI[res_trend_NDVI[sta + n]];
						}
						float sig, p;
						float y2[4] = { 0 };
						float u[4] = { 0 };
						for (int n = 1; n< 3; n++)
						{
							sig = (x[n] - x[n - 1]) / (x[n + 1] - x[n - 1]);
							p = sig*y2[n - 1] + 2;
							y2[n] = (sig - 1) / p;
							u[n] = (y[n + 1] - y[n]) / (x[n + 1] - x[n]) - (y[n] - y[n - 1]) / (x[n] - x[n - 1]);
							u[n] = (6.0*u[n] / (x[n + 1] - x[n - 1]) - sig*u[n - 1]) / p;
						}
						for (int n = 3; n >= 0; n--)
							y2[n] = y2[n] * y2[n + 1] + u[n];

						int klo = 0;
						int khi = 3;
						while (khi - klo > 1)            
						{
							int k = (khi + klo) >> 1;
							if (x[k] > nores_trend_NDVI[m])
								khi = k;
							else klo = k;
						}
						float h = x[khi] - x[klo];
						float a = (x[khi] - nores_trend_NDVI[m]) / h;
						float b = (nores_trend_NDVI[m] - x[klo]) / h;
						trend_NDVI[nores_trend_NDVI[m]] = a*y[klo] + b*y[khi] + ((a*a*a - a)*y2[klo] + (b*b*b - b)*y2[khi])*h*h / 6.0;
					}
					indic = 1;
				}
				else
					indic = 0;
			}
			else
				indic = 0;

			//STSG
			if (indic == 1)
			{
				if (snow_address == 1)
				{
					//processing contaminated NDVI by snow
					int count_vector_QA = 0;
					for (int k = 0; k < n_B; k++)
					{
						if (img_QA[i + (j + Buffer_Up) *n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 2)
							count_vector_QA++;
					}
					if (count_vector_QA != 0)
					{
						int bv_count = 0;
						float bv_total = 0.;
						for (int yeari = 0; yeari < n_Years; yeari++)
						{
							for (int k = 0; k <6; k++)
							{
								if (img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + yeari*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 0 || img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + yeari*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 1)
								{
									bv_count++;
									bv_total = bv_total + img_NDVI[i + (j + Buffer_Up) *n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + yeari*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B];
								}
							}
						}
						if (bv_count != 0)
						{
							float bv = bv_total / bv_count;
							for (int k = 0; k < n_B; k++)
							{
								if (img_QA[i + (j + Buffer_Up) *n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 2)
								{
									vector_in[k] = bv;
									trend_NDVI[k] = bv;
								}
							}
						}
					}
				}

				//calculating the weight for each point
				float gdis = 0.0;
				float *fl = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 2];
				int count_fl = 0;
				float mean_fl = 0;
				for (int k = 0; k < n_B; k++)
				{
					if (img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 0||img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 1)
					{
						fl[k] = vector_in[k] - trend_NDVI[k];
						count_fl++;
						mean_fl += fl[k];
					}
					else
						fl[k] = -1.0;
				}
				if (count_fl != 0)
					mean_fl = mean_fl / count_fl;
				for (int k = 0; k < n_B; k++)
				{
					if (fl[k] == -1.0)
						fl[k] = mean_fl;
				}

				for (int k = 0; k < n_B; k++)
				{
					float min_fl = 0;
					float max_fl = 0;
					for (int k = 0; k < n_B; k++)
					{
						if (min_fl > fl[k])
							min_fl = fl[k];
						if (max_fl < fl[k])
							max_fl = fl[k];
					}
					fl[k] = (fl[k] - min_fl) / (max_fl - min_fl);
					if ((vector_in[k] - trend_NDVI[k]) >= 0)
						gdis = gdis + fl[k] * (vector_in[k] - trend_NDVI[k]);
					else
						gdis = gdis + fl[k] * (trend_NDVI[k] - vector_in[k]);
				}

				for (int k = 0; k < n_B; k++)
				{
					if (img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 0)
						trend_NDVI[k] = vector_in[k];
					if (img_QA[i + (j + Buffer_Up) *n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] != 0 && img_QA[i + (j + Buffer_Up) *n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] != 1)
						vector_in[k] = trend_NDVI[k];
				}

				float *vec_fil = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 3];
				float ormax = gdis;
				int loop_times = 1;
				while (gdis <= ormax && loop_times < 50)
				{
					loop_times = loop_times + 1;
					for (int k = 0; k < n_B; k++)
						vec_fil[k] = trend_NDVI[k];
					//The Savitzky-Golay fitting; set the window width(4, 4) and degree(6) for repetition
					double savgolFilter[] = { -0.00543880, 0.0435097, -0.152289, 0.304585, 0.619267, 0.304585, -0.152289, 0.0435097, -0.00543880 };
					int savgolFilterW = sizeof(savgolFilter) / sizeof(savgolFilter[0]);
					int ra4W = n_B;
					float *new_ra4 = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 3 + n_B];
					for (int ii = 0; ii < savgolFilterW + ra4W - 1; ii++)
					{
						if (ii < (savgolFilterW - 1) / 2)
							new_ra4[ii] = ((vector_in[0] >= trend_NDVI[0]) ? vector_in[0] : trend_NDVI[0]);
						else if (ii >((savgolFilterW - 1) / 2 + ra4W - 1))
							new_ra4[ii] = ((vector_in[ra4W - 1] >= trend_NDVI[ra4W - 1]) ? vector_in[ra4W - 1] : trend_NDVI[ra4W - 1]);
						else
							new_ra4[ii] = ((vector_in[ii - (savgolFilterW - 1) / 2] >= trend_NDVI[ii - (savgolFilterW - 1) / 2]) ? vector_in[ii - (savgolFilterW - 1) / 2] : trend_NDVI[ii - (savgolFilterW - 1) / 2]);
					}
					for (int ii = 0; ii < ra4W; ii++)
					{
						float temp = 0;
						for (int jj = 0; jj < savgolFilterW; jj++)
							temp += savgolFilter[jj] * new_ra4[ii + jj];
						trend_NDVI[ii] = temp;
					}
					ormax = gdis;
					//calculating the fitting-effect index
					gdis = 0.0;
					for (int k = 0; k < n_B; k++)
					{
						if ((vector_in[k] - trend_NDVI[k]) >= 0)
							gdis = gdis + fl[k] * (vector_in[k] - trend_NDVI[k]);
						else
							gdis = gdis + fl[k] * (trend_NDVI[k] - vector_in[k]);
					}
				}

				for (int k = 0; k < n_B; k++)
					vector_out[i + (j + StartY)*n_X + k*n_X*TotalY + y*n_X*TotalY*n_B] = vec_fil[k];
				for (int smi = 0; smi < n_B - 4; smi++)
				{
					float a1 = vec_fil[smi];
					float a2 = vec_fil[smi + 1];
					float a3 = vec_fil[smi + 2];
					float a4 = vec_fil[smi + 3];
					float a5 = vec_fil[smi + 4];
					if ((a1 > a2) && (a2 < a3) && (a3 > a4) && (a4 < a5))
					{
						vector_out[i + (j + StartY)*n_X + (smi + 1) * n_X*TotalY + y*n_X*TotalY*n_B] = (a1 + a3) / 2.0;
						vector_out[i + (j + StartY)*n_X + (smi + 3) * n_X*TotalY + y*n_X*TotalY*n_B] = (a3 + a5) / 2.0;
					}
				}
			}

			//SG
			if (indic == 0)
			{
				if (snow_address == 1)
				{
					//processing contaminated NDVI by snow
					int count_vector_QA = 0;
					for (int k = 0; k < n_B; k++)
					{
						if (img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 2)
							count_vector_QA++;
					}
					if (count_vector_QA != 0)
					{
						int bv_count = 0;
						float bv_total = 0.;
						for (int yeari = 0; yeari < n_Years; yeari++)
						{
							for (int k = 0; k < 6; k++)
							{
								if (img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + yeari*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 0 || img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + yeari*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 1)
								{
									bv_count++;
									bv_total = bv_total + img_NDVI[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + yeari*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B];
								}
							}
						}
						if (bv_count != 0)
						{
							float bv = bv_total / bv_count;
							for (int k = 0; k < n_B; k++)
							{
								if (img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] == 2)
								{
									vector_in[k] = bv;
									trend_NDVI[k] = bv;
								}
							}
						}
					}
				}

				int count_vector_QA = 0;
				int *res_vector_QA = &d_index[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 3 + (2 * win + 1)*(2 * win + 1) * 2];
				for (int k = 0; k < n_B; k++)
				{
					if (img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] <= 2 && img_QA[i + (j + Buffer_Up)*n_X + k*n_X*(n_Y + Buffer_Up + Buffer_Dn) + y*n_X*(n_Y + Buffer_Up + Buffer_Dn)*n_B] != -1)
						res_vector_QA[count_vector_QA++] = k;
				}
				if (count_vector_QA < n_B&&count_vector_QA>1)
				{
					double k;
					int x = 0;
					int l = 0;
					if (res_vector_QA[count_vector_QA - 1] != n_B - 1)
					{
						k = (vector_in[res_vector_QA[count_vector_QA - 1]] - vector_in[res_vector_QA[count_vector_QA - 2]]) / (res_vector_QA[count_vector_QA - 1] - res_vector_QA[count_vector_QA - 2]);
						vector_in[n_B - 1] = vector_in[res_vector_QA[count_vector_QA - 1]] + k*(n_B - 1 - res_vector_QA[count_vector_QA - 1]);
						res_vector_QA[count_vector_QA++] = n_B - 1;
					}
					int count_res_vector_QA = count_vector_QA;
					while (count_vector_QA < n_B&&x < count_res_vector_QA - 1)
					{
						l = res_vector_QA[x + 1] - res_vector_QA[x];
						int n = 1;
						while (l > 1)
						{
							k = (vector_in[res_vector_QA[x + 1]] - vector_in[res_vector_QA[x]]) / (res_vector_QA[x + 1] - res_vector_QA[x]);
							vector_in[res_vector_QA[x] + n] = vector_in[res_vector_QA[x]] + k*n;
							count_vector_QA++;
							n++;
							l--;
						}
						x++;
					}
					if (res_vector_QA[0] != 0)
					{
						k = (vector_in[res_vector_QA[1]] - vector_in[res_vector_QA[0]]) / (res_vector_QA[1] - res_vector_QA[0]);
						l = res_vector_QA[0];
						int n = 0;
						do
						{
							vector_in[n] = vector_in[res_vector_QA[0]] - k*l;
							n++;
							l--;
						} while (l >= 1);
					}
				}

				float* rst = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 3];
				//set the window width(4, 4) and degree(2) for computing trend curve
				double savgolFilter[] = { -0.0909091, 0.0606061, 0.168831, 0.233766, 0.255411, 0.233766, 0.168831, 0.0606061, -0.0909091 };
				int savgolFilterW = sizeof(savgolFilter) / sizeof(savgolFilter[0]);
				int vector_inW = n_B;
				float *new_vector_in = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 3 + n_B];
				for (int ii = 0; ii < savgolFilterW + vector_inW - 1; ii++)
				{
					if (ii < (savgolFilterW - 1) / 2)
						new_vector_in[ii] = vector_in[0];
					else if (ii >((savgolFilterW - 1) / 2 + vector_inW - 1))
						new_vector_in[ii] = vector_in[vector_inW - 1];
					else
						new_vector_in[ii] = vector_in[ii - (savgolFilterW - 1) / 2];
				}
				for (int ii = 0; ii < vector_inW; ii++)
				{
					float temp = 0;
					for (int jj = 0; jj < savgolFilterW; jj++)
						temp += savgolFilter[jj] * new_vector_in[ii + jj];
					rst[ii] = temp;
				}

				//Calculating the weight for each point
				float gdis = 0.0;
				float *fl = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 2];
				float maxdif = 0;
				for (int k = 0; k < n_B; k++)
				{
					if ((vector_in[k] - rst[k]) >= 0)
						fl[k] = vector_in[k] - rst[k];
					else
						fl[k] = rst[k] - vector_in[k];
					if (k == 0)
						maxdif = fl[k];
					else
					{
						if (maxdif < fl[k])
							maxdif = fl[k];
					}
				}
				for (int k = 0; k < n_B; k++)
				{
					if (vector_in[k] >= rst[k])
					{
						fl[k] = 1.0;
						gdis = gdis + fl[k] * (vector_in[k] - rst[k]);
					}
					else
					{
						fl[k] = 1 - (rst[k] - vector_in[k]) / maxdif;
						gdis = gdis + fl[k] * (rst[k] - vector_in[k]);
					}
				}

				float ormax = gdis;
				int loop_times = 1;
				while (gdis <= ormax && loop_times < 15)
				{
					loop_times = loop_times + 1;
					for (int k = 0; k < n_B; k++)
						vector_out[i + (j + StartY)*n_X + k*n_X*TotalY + y*n_X*TotalY*n_B] = rst[k];
					//The Savitzky-Golay fitting; set the window width(4, 4) and degree(6) for repetition
					double savgolFilter[] = { -0.00543880, 0.0435097, -0.152289, 0.304585, 0.619267, 0.304585, -0.152289, 0.0435097, -0.00543880 };
					int savgolFilterW = sizeof(savgolFilter) / sizeof(savgolFilter[0]);
					int ra4W = n_B;
					float *new_ra4 = &d_res[(i + j*n_X)*(2 * win + 1)*(2 * win + 1) * 4 + (2 * win + 1)*(2 * win + 1) * 3 + n_B];
					for (int ii = 0; ii < savgolFilterW + ra4W - 1; ii++)
					{
						if (ii < (savgolFilterW - 1) / 2)
							new_ra4[ii] = (vector_in[0] >= rst[0]) ? vector_in[0] : rst[0];
						else if (ii >((savgolFilterW - 1) / 2 + ra4W - 1))
							new_ra4[ii] = (vector_in[ra4W - 1] >= rst[ra4W - 1]) ? vector_in[ra4W - 1] : rst[ra4W - 1];
						else
							new_ra4[ii] = (vector_in[ii - (savgolFilterW - 1) / 2] >= rst[ii - (savgolFilterW - 1) / 2]) ? vector_in[ii - (savgolFilterW - 1) / 2] : rst[ii - (savgolFilterW - 1) / 2];
					}
					for (int ii = 0; ii < ra4W; ii++)
					{
						float temp = 0;
						for (int jj = 0; jj < savgolFilterW; jj++)
							temp += savgolFilter[jj] * new_ra4[ii + jj];
						rst[ii] = temp;
					}
					ormax = gdis;
					//Calculating the fitting-effect index
					gdis = 0.0;
					for (int k = 0; k < n_B; k++)
					{
						if ((vector_in[k] - rst[k]) >= 0)
							gdis = gdis + fl[k] * (vector_in[k] - rst[k]);
						else
							gdis = gdis + fl[k] * (rst[k] - vector_in[k]);
					}
				}
			}
		}
	}
}