#pragma once

#ifndef FILTER_H
#define FILTER_H

void STSG_filter(int win, float sampcorr, float *img_NDVI, float *img_QA, float *reference_data, int ns, int nl, int nb, int ny, int snow_address, float *vector_out);

#endif // !FILTER_H
