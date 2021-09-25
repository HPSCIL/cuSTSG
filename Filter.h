#pragma once

#ifndef FILTER_H
#define FILTER_H

__global__ void Short_to_Float(short *imgNDVI, unsigned char *imgQA, int n_X, int n_Y, int n_B, int n_Years, float *img_NDVI, float *img_QA);
__global__ void Generate_NDVI_reference(float cosyear, int win_year, float *img_NDVI, float *img_QA, int n_X, int n_Y, int n_B, int n_Years, float *NDVI_reference, float *d_res_3, int *d_res_vec_res1);
__global__ void Compute_d_res(float *img_NDVI, float *img_QA, float *reference_data, int StartY, int TotalY, int Buffer_Up, int Buffer_Dn, int n_X, int n_Y, int n_B, int n_Years, int win, float *d_res);
__global__ void STSG_filter(float *img_NDVI, float *img_QA, float *reference_data, int StartY, int TotalY, int Buffer_Up, int Buffer_Dn, int n_X, int n_Y, int n_B, int n_Years, int win, float sampcorr, int snow_address, float *vector_out, float *d_vector_in, float *d_res, float *d_res_3, int *d_index);

#endif // !FILTER_H
