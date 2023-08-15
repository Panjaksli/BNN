#include "NNet_samples.h"
namespace BNN {
	NNet Down_conv(dim1<3> idims, idx factor) {
		if(factor <= 0) factor = 1;
		vector<Layer*> top;
		top.push_back(new Input(idims));
		top.push_back(new Conv(idims[0], factor, factor, 0, top.back()));
		top.push_back(new Output(top.back()));
		return NNet(top, new Optimizer(), "Downscl_conv");
	}
	NNet Upsc_conv(dim1<3> idims, idx factor) {
		if(factor <= 0) factor = 1;
		vector<Layer*> top;
		top.push_back(new Input(idims));
		top.push_back(new TConv(idims[0], factor, factor, 0, top.back()));
		top.push_back(new Output(top.back()));
		return NNet(top, new Optimizer(), "Upscl_conv");
	}
	NNet Downscaler(dim1<3> idims, idx factor) {
		if(factor <= 0) factor = 1;
		vector<Layer*> top;
		top.push_back(new Input(idims));
		top.push_back(new AvgPool(factor, factor, 0, top.back()));
		top.push_back(new Output(top.back()));
		return NNet(top, new Optimizer(), "Downscaler");
	}
	NNet Downsampler(dim1<3> idims, idx factor) {
		if(factor <= 0) factor = 1;
		vector<Layer*> top;
		top.push_back(new Input(idims));
		top.push_back(new AvgPool(factor, factor, 0, top.back()));
		top.push_back(new AvgUpool(factor, factor, 0, top.back()));
		top.push_back(new Output(top.back()));
		return NNet(top, new Optimizer(), "Downsampler");
	}
	NNet Upscaler(dim1<3> idims, idx factor) {
		if(factor <= 0) factor = 1;
		vector<Layer*> top;
		top.push_back(new Input(idims));
		top.push_back(new AvgUpool(factor, factor, 0, top.back()));
		top.push_back(new Output(top.back()));
		return NNet(top, new Optimizer(), "Upscaler");
	}
}