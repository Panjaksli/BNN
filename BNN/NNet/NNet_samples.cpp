#include "NNet_samples.h"
namespace BNN {
	NNet Down_conv(dim1<3> idims, int factor) {
		if(factor <= 0) factor = 1;
		vector<Layer*> top;
		Input* inp = new Input(idims);
		top.push_back(new Conv(idims[0],factor, factor, 0, inp));
		Output* out = new Output(top.back());
		return NNet(inp, top, out, new Optimizer(), "Downscaler");
	}
	NNet Upsc_conv(dim1<3> idims, int factor) {
		if(factor <= 0) factor = 1;
		vector<Layer*> top;
		Input* inp = new Input(idims);
		top.push_back(new TConv(idims[0], factor, factor, 0, inp));
		Output* out = new Output(top.back());
		return NNet(inp, top, out, new Optimizer(), "Upscaler");
	}
	NNet Downscaler(dim1<3> idims, int factor) {
		if(factor <= 0) factor = 1;
		vector<Layer*> top;
		Input* inp = new Input(idims);
		top.push_back(new AvgPool(factor, factor, 0, inp));
		Output* out = new Output(top.back());
		return NNet(inp, top, out, new Optimizer(), "Downscaler");
	}
	NNet Downsampler(dim1<3> idims, int factor) {
		if(factor <= 0) factor = 1;
		vector<Layer*> top;
		Input* inp = new Input(idims);
		top.push_back(new AvgPool(factor, factor, 0, inp));
		top.push_back(new AvgUpool(factor, factor, 0, top.back()));
		Output* out = new Output(top.back());
		return NNet(inp, top, out, new Optimizer(), "Downsampler");
	}
	NNet Upscaler(dim1<3> idims, int factor) {
		if(factor <= 0) factor = 1;
		vector<Layer*> top;
		Input* inp = new Input(idims);
		top.push_back(new AvgUpool(factor, factor, 0, inp));
		Output* out = new Output(top.back());
		return NNet(inp, top, out, new Optimizer(), "Upscaler");
	}
}