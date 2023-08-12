#pragma once
#include "NNet.h"
namespace BNN {
	NNet Downscaler(dim1<3> idims, int factor);
	NNet Downsampler(dim1<3> idims, int factor);
	NNet Down_conv(dim1<3> idims, int factor);
	NNet Upscaler(dim1<3> idims, int factor);
	NNet Upsc_conv(dim1<3> idims, int factor);
}