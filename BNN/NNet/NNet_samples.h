#pragma once
#include "NNet.h"
namespace BNN {
	NNet Downscaler(dim1<3> idims, idx factor);
	NNet Downsampler(dim1<3> idims, idx factor);
	NNet Down_conv(dim1<3> idims, idx factor);
	NNet Upscaler(dim1<3> idims, idx factor);
	NNet Upsc_conv(dim1<3> idims, idx factor);
}