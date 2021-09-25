**cuSTSG**

Version 1.0

**Overview**

High-quality Normalized Difference Vegetation Index (NDVI) time-series data are important for many applications. To generate high-quality NDVI time series, a noise-reduction method integrating spatial-temporal information with Savitzky-Golay filter (named STSG) was proposed by Cao et al. (2018). STSG assumes discontinuous clouds in space, and employs neighboring pixels in the noise reduction for a target pixel in a particular year. The relationship between the NDVI of the target pixel and those of the neighboring pixels is obtained from the multi-year NDVI time series. STSG is able to address the problem of temporally continuous NDVI gaps and effectively increases local low NDVI values without overcorrecting. However, STSG largely depends on the quality flags of NDVI time-series data, and inaccurate quality flags will greatly deviate the production of the target and neighboring pixels from the ground-truth data. STSG also requires extensive computing time when dealing with large-scale applications.

To address the above issues of STSG, we designed and implemented a GPU-enabled STSG program based on the Compute Unified Device Architecture (CUDA), called cuSTSG. Firstly, the cosine similarities between the annual NDVI time series are used to identify and exclude the NDVI values with inaccurate quality flags from the NDVI seasonal growth trajectory. Secondly, the computational performance is improved by reducing redundant computations, and parallelizing the computationally intensive procedures using CUDA on GPUs. The experiments showed that cuSTSG effectively mitigated the effects of inaccurate quality flags on the final production, and improved the accuracy by 0.11 times, compared with the original STSG. The results also showed cuSTSG achieved a speed-up over 30 on a GPU, compared with the C++-implemented STSG on a CPU.

**Key features of cuSTSG**

- Supports a wide range of CUDA-enabled GPUs (https://developer.nvidia.com/cuda-gpus)
  - Automatic setting of the numbers of threads and thread blocks according to the GPU&#39;s available computing resources (e.g., memory, streaming multiprocessors, and warp)
  - Adaptive cyclic task assignment to achieve better load balance
  - Adaptive data domain decomposition when the size of images and temporary products exceeds the GPU&#39;s memory
  - All above are completely transparent to users
- Intakes more than two years of MODIS images as the input
- Supports both Windows and Linux/Unix operating systems

**References**

- Cao, R., Chen, Y., Shen, M., Chen, J., Zhou, J., Wang, C., Yang, W., 2018. A simple method to improve the quality of NDVI time-series data by integrating spatiotemporal information with the Savitzky-Golay filter. Remote Sensing of Environment 217, 244–257.
- Chen, J., Jönsson, Per., Tamura, M., Gu, Z., Matsushita, B., Eklundh, L., 2004. A simple method for reconstructing a high-quality NDVI time-series data set based on the Savitzky–Golay filter. Remote Sensing of Environment 91, 332–344.

**To Cite cuSTSG in Publications**

- A paper describing cuSTSG will be submitted to a scientific journal for publication soon
- For now, you may just cite the URL of the source codes of cuSTSG (https://github.com/HPSCIL/cuSTSG) in your publications

**Compilation**

- Requirements:
  - A computer with a CUDA-enabled GPU (https://developer.nvidia.com/cuda-gpus)
  - A C/C++ compiler (e.g., Microsoft Visual Studio for Windows, and gcc/g++ for Linux/Unix) installed and tested
  - Nvidia CUDA Toolkit (https://developer.nvidia.com/cuda-downloads) installed and tested
  - Geospatial Data Abstraction Library (GDAL, http://gdal.org) installed and tested
- For the Windows operating system (using MS Visual Studio as an example)

1. Open all the source codes in Visual Studio
2. Click menu Project -\&gt; Properties -\&gt; VC++ Directories -\&gt; Include Directories, and add the &quot;include&quot; directory of GDAL (e.g., C:\GDAL\include)
3. Click menu Project -\&gt; Properties -\&gt; VC++ Directories -\&gt; Lib Directories, and add the &quot;lib&quot; directory of GDAL (e.g., C:\GDAL\lib)
4. Click menu Build -\&gt; Build Solution
5. Once successfully compiled, an executable file, cuSTSG.exe, is created.

- For the Linux/Unix operating system (using the CUDA compiler --- nvcc)

In a Linux/Unix terminal, type in:

1. $ cd /the-directory-of-source-codes/
2. $ nvcc -o cuSTSG cuSTSG.cu Filter.cu -lgdal
3. Once successfully compiled, an executable file, cuSTSG, is created.

**Usage**

- Before running the program, make sure that all MODIS images (the NDVI images and the quality flag images) have been pre-processed. They must have:
  - the same spatial and temporal resolution (e.g., 1-km and 16-day for MOD13A2)
  - the same image size (i.e., numbers of rows, columns, and bands)
  - the same map projection
- A text file must be manually created to specify the input and output images, and other parameters for cuSTSG.

Example (// for comments):

//\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

//Input parameters

//the years of the data (&quot;,&quot; is necessary to separate data)

Years = 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018

//the path of the NDVI data

NDVI\_path = ../TestData/NDVI/NDVI\_test\_

//the path of the NDVI quality flags

Reliability\_path = ../TestData/Reliability/Reliability\_test\_

//the path of the production

STSG\_Test\_path = ../TestData/STSG\_Test.tif

//the thereshold of cosine similarity to define similar years

cosyear = 0.90

//the half size of the window within which to search pixels with inaccurate quality flags in the dissimilar year

win\_year = 2

//the half size of the neighboring window within which to search similar pixels

win = 10

//the thereshold of correlation coefficient to define similar pixels

sampcorr = 0.9

//snow\_address indicates whether to deal with snow contamianted NDVI values(1 = yes / 0 = no)

snow\_address = 1

//\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

- The program runs as a command line. You may use the Command (i.e., cmd) in Windows, or a terminal in Linux/Unix.
  - For the Windows version:

$ cuSTSG.exe parameters.txt

  - For the Linux/Unix version:

$ ./cuSTSG parameters.txt

- Note: The computational performance of cuSTSG largely depends on the GPU. The more powerful the GPU is, the better performance.