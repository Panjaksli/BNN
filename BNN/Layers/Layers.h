#pragma once
#include "Layer.h"
#include "Input.h"
#include "Output.h"
#include "OutShuf.h"
#include "Conv.h"
#include "AvgPool.h"
#include "AvgUpool.h"
#include "Dense.h"
#include "TConv.h"
#include "SConv.h"
#include "Dropout.h"
#include "PixShuf.h"
#include "Resize.h"
#include "Shape.h"
namespace BNN {
	inline Layer* Layer_load(std::istream& in) {
		std::string token;
		in >> token;
		//I'm sure there is a better way, but whatever
		if(token == "Input")
			return Input::load(in);
		else if(token == "Output")
			return Output::load(in);
		else if(token == "OutShuf")
			return OutShuf::load(in);
		else if(token == "AvgPool")
			return AvgPool::load(in);
		else if(token == "AvgUpool")
			return AvgUpool::load(in);
		else if(token == "Dense")
			return Dense::load(in);
		else if(token == "Conv")
			return Conv::load(in);
		else if(token == "TConv")
			return TConv::load(in);
		else if(token == "Dropout")
			return Dropout::load(in);
		else if(token == "PixShuf")
			return PixShuf::load(in);
		else if(token == "Resize")
			return Resize::load(in);
		else if(token == "SConv")
			return SConv::load(in);
		else if(token == "Shape")
			return Shape::load(in);
		else return nullptr;
	}
}