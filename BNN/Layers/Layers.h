#pragma once
#include "Layer.h"
#include "Input.h"
#include "Output.h"
#include "Conv.h"
#include "AvgPool.h"
#include "AvgUpool.h"
#include "Dense.h"
#include "TConv.h"
#include "Dropout.h"
namespace BNN {
	inline Layer* Hidden_load(std::istream &in) {
		std::string token;
		in >> token;
		if(token == "AvgPool")
			return AvgPool::load(in);
		else if(token == "AvgUpool")
			return AvgUpool::load(in);
		else if(token == "Conv")
			return Conv::load(in);
		else if(token == "TConv")
			return TConv::load(in);
		else if(token == "Dropout")
			return Dropout::load(in);
		else if(token == "Dense")
			return Dense::load(in);
		else return nullptr;
	}
}