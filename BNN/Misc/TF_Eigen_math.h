#pragma once
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
namespace Eigen {
	template <typename Input, typename Kernel,
		typename OutputKernel = const NoOpOutputKernel>
	EIGEN_DEVICE_FUNC
		EIGEN_ALWAYS_INLINE static const typename internal::conditional <
		internal::traits<Input>::Layout == ColMajor,
		TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index,
		internal::traits<Input>::NumDimensions>,
		const TensorContractionOp<
		const array<IndexPair<typename internal::traits<Input>::Index>,
		1>,
		const TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index, 2>,
		const Kernel>,
		const TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index, 2>,
		const TensorImagePatchOp<Dynamic, Dynamic, const Input> >,
		const OutputKernel> >,
		TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index,
		internal::traits<Input>::NumDimensions>,
		const TensorContractionOp<
		const array<IndexPair<typename internal::traits<Input>::Index>,
		1>,
		const TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index, 2>,
		const TensorImagePatchOp<Dynamic, Dynamic, const Input> >,
		const TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index, 2>,
		const Kernel>,
		const OutputKernel> > > ::type
		SpatialConvolution(const Input& input, const Kernel& kernel,
			const Index row_stride = 1, const Index col_stride = 1,
			const PaddingType padding_type = PADDING_SAME,
			const Index row_in_stride = 1,
			const Index col_in_stride = 1,
			const OutputKernel& output_kernel = OutputKernel()) {
		typedef typename internal::traits<Input>::Index TensorIndex;
		TensorRef<Tensor<typename internal::traits<Input>::Scalar,
			internal::traits<Input>::NumDimensions,
			internal::traits<Input>::Layout, TensorIndex> >
			in(input);
		TensorRef<Tensor<typename internal::traits<Kernel>::Scalar,
			internal::traits<Kernel>::NumDimensions,
			internal::traits<Kernel>::Layout, TensorIndex> >
			kern(kernel);

		EIGEN_STATIC_ASSERT(
			internal::traits<Input>::Layout == internal::traits<Kernel>::Layout,
			YOU_MADE_A_PROGRAMMING_MISTAKE);
		const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

		const int NumDims = internal::traits<Input>::NumDimensions;

		// Number of filters to apply. This is the same as the output depth of the
		// result
		const TensorIndex kernelFilters =
			isColMajor ? kern.dimensions()[0] : kern.dimensions()[3];
		// Number of channels. This is the same as the input depth.
		const TensorIndex kernelChannels =
			isColMajor ? kern.dimensions()[1] : kern.dimensions()[2];
		const TensorIndex kernelRows =
			isColMajor ? kern.dimensions()[2] : kern.dimensions()[1];
		const TensorIndex kernelCols =
			isColMajor ? kern.dimensions()[3] : kern.dimensions()[0];

		const Index kernelRowsEff =
			kernelRows + (kernelRows - 1) * (row_in_stride - 1);
		const Index kernelColsEff =
			kernelCols + (kernelCols - 1) * (col_in_stride - 1);

		array<IndexPair<TensorIndex>, 1> contract_dims;
		contract_dims[0] = IndexPair<TensorIndex>(1, 0);

		const TensorIndex InputRows =
			isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
		const TensorIndex InputCols =
			isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);

		TensorIndex out_height;
		TensorIndex out_width;
		switch(padding_type) {
			case PADDING_VALID:
				out_height = numext::ceil((InputRows - kernelRowsEff + 1.f) /
					static_cast<float>(row_stride));
				out_width = numext::ceil((InputCols - kernelColsEff + 1.f) /
					static_cast<float>(col_stride));
				break;
			case PADDING_SAME:
				out_height = numext::ceil(InputRows / static_cast<float>(row_stride));
				out_width = numext::ceil(InputCols / static_cast<float>(col_stride));
				break;
			default:
				// Initialize unused variables to avoid a compiler warning
				out_height = 0;
				out_width = 0;
				eigen_assert(false && "unexpected padding");
		}

		// Molds the output of the patch extraction code into a 2d tensor:
		// - the first dimension (dims[0]): the patch values to be multiplied with the
		// kernels
		// - the second dimension (dims[1]): everything else
		DSizes<TensorIndex, 2> pre_contract_dims;
		if(isColMajor) {
			pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
			pre_contract_dims[1] = out_height * out_width;
			for(int i = 3; i < NumDims; ++i) {
				pre_contract_dims[1] *= in.dimension(i);
			}
		}
		else {
			pre_contract_dims[1] = kernelChannels * kernelRows * kernelCols;
			pre_contract_dims[0] = out_height * out_width;
			for(int i = 0; i < NumDims - 3; ++i) {
				pre_contract_dims[0] *= in.dimension(i);
			}
		}

		// Molds the output of the contraction into the shape expected by the used
		// (assuming this is ColMajor):
		// - 1st dim: kernel filters
		// - 2nd dim: output height
		// - 3rd dim: output width
		// - 4th dim and beyond: everything else including batch size
		DSizes<TensorIndex, NumDims> post_contract_dims;
		if(isColMajor) {
			post_contract_dims[0] = kernelFilters;
			post_contract_dims[1] = out_height;
			post_contract_dims[2] = out_width;
			for(int i = 3; i < NumDims; ++i) {
				post_contract_dims[i] = in.dimension(i);
			}
		}
		else {
			post_contract_dims[NumDims - 1] = kernelFilters;
			post_contract_dims[NumDims - 2] = out_height;
			post_contract_dims[NumDims - 3] = out_width;
			for(int i = 0; i < NumDims - 3; ++i) {
				post_contract_dims[i] = in.dimension(i);
			}
		}

		DSizes<TensorIndex, 2> kernel_dims;
		if(isColMajor) {
			kernel_dims[0] = kernelFilters;
			kernel_dims[1] = kernelChannels * kernelRows * kernelCols;
		}
		else {
			kernel_dims[0] = kernelChannels * kernelRows * kernelCols;
			kernel_dims[1] = kernelFilters;
		}
		// TODO(yangke): choose() is defined in TensorContraction.h -- consider
		// moving it to somewhere more "common".
		return choose(
			Cond<internal::traits<Input>::Layout == ColMajor>(),
			kernel.reshape(kernel_dims)
			.contract(input
			.extract_image_patches(
			kernelRows, kernelCols, row_stride, col_stride,
			row_in_stride, col_in_stride, padding_type)
			.reshape(pre_contract_dims),
			contract_dims, output_kernel)
			.reshape(post_contract_dims),
			input
			.extract_image_patches(kernelRows, kernelCols, row_stride, col_stride,
			row_in_stride, col_in_stride, padding_type)
			.reshape(pre_contract_dims)
			.contract(kernel.reshape(kernel_dims), contract_dims, output_kernel)
			.reshape(post_contract_dims));
	}

#ifdef EIGEN_HAS_INDEX_LIST
	typedef IndexList<type2index<0>, type2index<0>, type2index<1>, type2index<1> >
		ReverseColMajor;
	typedef IndexList<type2index<1>, type2index<1>, type2index<0>, type2index<0> >
		ReverseRowMajor;
#else
	typedef array<bool, 4> ReverseColMajor;
	typedef array<bool, 4> ReverseRowMajor;
#endif

	template <typename OutputBackward, typename Kernel>
	EIGEN_ALWAYS_INLINE static const typename internal::conditional <
		internal::traits<OutputBackward>::Layout == ColMajor,
		TensorReshapingOp<
		const DSizes<typename internal::traits<OutputBackward>::Index,
		internal::traits<OutputBackward>::NumDimensions>,
		const TensorContractionOp<
		const array<
		IndexPair<typename internal::traits<OutputBackward>::Index>, 1>,
		const Eigen::TensorForcedEvalOp<const TensorReshapingOp<
		const DSizes<typename internal::traits<OutputBackward>::Index,
		2>,
		const TensorShufflingOp<
		const array<
		typename internal::traits<OutputBackward>::Index, 4>,
		const TensorReverseOp<const ReverseColMajor,
		const Kernel> > > >,
		const TensorReshapingOp<
		const DSizes<typename internal::traits<OutputBackward>::Index,
		2>,
		const TensorImagePatchOp<Dynamic, Dynamic,
		const OutputBackward> > > >,
		TensorReshapingOp<
		const DSizes<typename internal::traits<OutputBackward>::Index,
		internal::traits<OutputBackward>::NumDimensions>,
		const TensorContractionOp<
		const array<
		IndexPair<typename internal::traits<OutputBackward>::Index>, 1>,
		const TensorReshapingOp<
		const DSizes<typename internal::traits<OutputBackward>::Index,
		2>,
		const TensorImagePatchOp<Dynamic, Dynamic,
		const OutputBackward> >,
		const Eigen::TensorForcedEvalOp<const TensorReshapingOp<
		const DSizes<typename internal::traits<OutputBackward>::Index,
		2>,
		const TensorShufflingOp<
		const array<
		typename internal::traits<OutputBackward>::Index, 4>,
		const TensorReverseOp<const ReverseRowMajor,
		const Kernel> > > > > > > ::type
		SpatialConvolutionBackwardInput(
			const Kernel& kernel, const OutputBackward& output_backward,
			typename internal::traits<OutputBackward>::Index inputRows,
			typename internal::traits<OutputBackward>::Index inputCols,
			const DenseIndex row_stride = 1, const DenseIndex col_stride = 1,
			const DenseIndex row_in_stride = 1, const DenseIndex col_in_stride = 1) {
		typedef typename internal::traits<OutputBackward>::Index TensorIndex;
		typedef typename internal::traits<OutputBackward>::Scalar OutScalar;
		TensorRef<Tensor<typename internal::traits<Kernel>::Scalar,
			internal::traits<Kernel>::NumDimensions,
			internal::traits<Kernel>::Layout, TensorIndex> >
			kern(kernel);
		TensorRef<Tensor<OutScalar, internal::traits<OutputBackward>::NumDimensions,
			internal::traits<OutputBackward>::Layout, TensorIndex> >
			out(output_backward);

		EIGEN_STATIC_ASSERT(internal::traits<Kernel>::Layout ==
			internal::traits<OutputBackward>::Layout,
			YOU_MADE_A_PROGRAMMING_MISTAKE);

		static const bool isColMajor =
			(internal::traits<OutputBackward>::Layout == ColMajor);

		static const int NumDims = internal::traits<OutputBackward>::NumDimensions;

		// Number of filters to apply. This is the same as the output depth of the
		// result
		const TensorIndex kernelFilters =
			isColMajor ? kern.dimensions()[0] : kern.dimensions()[3];
		// Number of channels. This is the same as the input depth.
		const TensorIndex kernelChannels =
			isColMajor ? kern.dimensions()[1] : kern.dimensions()[2];
		const TensorIndex kernelRows =
			isColMajor ? kern.dimensions()[2] : kern.dimensions()[1];
		const TensorIndex kernelCols =
			isColMajor ? kern.dimensions()[3] : kern.dimensions()[0];

		// This is the effective kernel size, taking into account the (*_in_stride -
		// 1) zero-values
		// inserted between consecutive kernel elements in atrous convolution
		const TensorIndex kernelRowsEff =
			kernelRows + (kernelRows - 1) * (row_in_stride - 1);
		const TensorIndex kernelColsEff =
			kernelCols + (kernelCols - 1) * (col_in_stride - 1);

		const TensorIndex outputRows = isColMajor
			? output_backward.dimension(1)
			: output_backward.dimension(NumDims - 2);
		const TensorIndex outputCols = isColMajor
			? output_backward.dimension(2)
			: output_backward.dimension(NumDims - 3);

		// Computing the forward padding
		const TensorIndex forward_pad_top = numext::maxi<Index>(
			0, ((outputRows - 1) * row_stride + kernelRowsEff - inputRows) / 2);
		const TensorIndex forward_pad_left = numext::maxi<Index>(
			0, ((outputCols - 1) * col_stride + kernelColsEff - inputCols) / 2);
		const TensorIndex padding_top = kernelRowsEff - 1 - forward_pad_top;
		const TensorIndex padding_left = kernelColsEff - 1 - forward_pad_left;

		const TensorIndex padding_bottom = inputRows - (outputRows - 1) * row_stride -
			2 - padding_top + kernelRowsEff;
		const TensorIndex padding_right = inputCols - (outputCols - 1) * col_stride -
			2 - padding_left + kernelColsEff;

		eigen_assert(padding_top >= 0);
		eigen_assert(padding_left >= 0);
		eigen_assert(padding_bottom >= 0);
		eigen_assert(padding_right >= 0);

		// The kernel has dimensions filters X channels X patch_rows X patch_cols
		// We need to reverse the kernel along dimensions corresponding to rows and
		// cols.
		// TODO(yangke): we can make things slightly faster by collapsing the
		// dimensions
		// where we don't reverse. Try that once we have a faster compiler.
		typedef typename internal::conditional<isColMajor, ReverseColMajor,
			ReverseRowMajor>::type Reverse;
		Reverse kernel_reverse;

#ifndef EIGEN_HAS_INDEX_LIST
		if(isColMajor) {
			kernel_reverse[0] = false;
			kernel_reverse[1] = false;
			kernel_reverse[2] = true;
			kernel_reverse[3] = true;
		}
		else {
			kernel_reverse[0] = true;
			kernel_reverse[1] = true;
			kernel_reverse[2] = false;
			kernel_reverse[3] = false;
		}
#endif

		// Reorder the dimensions to:
		//   filters x patch_rows x patch_cols x channels
		array<TensorIndex, 4> kernel_shuffle;
		if(isColMajor) {
			//  From: filters x channels x rows x cols
			//  To:   filters x rows x cols x channels
			kernel_shuffle[0] = 0;
			kernel_shuffle[1] = 2;
			kernel_shuffle[2] = 3;
			kernel_shuffle[3] = 1;
		}
		else {
			//  From: cols x rows x channels x filters
			//  To:   channels x cols x rows x filters
			kernel_shuffle[0] = 2;
			kernel_shuffle[1] = 0;
			kernel_shuffle[2] = 1;
			kernel_shuffle[3] = 3;
		}

		// Collapse the dims
		DSizes<TensorIndex, 2> kernel_dims;
		if(isColMajor) {
			kernel_dims[0] = kernelFilters * kernelRows * kernelCols;
			kernel_dims[1] = kernelChannels;
		}
		else {
			kernel_dims[1] = kernelFilters * kernelRows * kernelCols;
			kernel_dims[0] = kernelChannels;
		}

		// The output_backward has dimensions out_depth X out_rows X out_cols X OTHERS
		// When we extract the image patches from output_backward, it will have
		// dimensions
		//   out_depth X (patch_rows * patch_cols) X (input_rows * input_cols *
		//   OTHERS)
		DSizes<TensorIndex, 2> pre_contract_dims;
		if(isColMajor) {
			pre_contract_dims[0] = kernelFilters * kernelRows * kernelCols;
			pre_contract_dims[1] = inputRows * inputCols;
			for(int i = 3; i < NumDims; ++i) {
				pre_contract_dims[1] *= out.dimension(i);
			}
		}
		else {
			pre_contract_dims[1] = kernelFilters * kernelRows * kernelCols;
			pre_contract_dims[0] = inputRows * inputCols;
			for(int i = 0; i < NumDims - 3; ++i) {
				pre_contract_dims[0] *= out.dimension(i);
			}
		}

		// We will contract along the collapsed dimension that contains the
		// kernelFilters, the kernelRows and the kernelCols.
		array<IndexPair<TensorIndex>, 1> contract_dims;
		if(isColMajor) {
			// col-major: kernel.contract(output.patches)
			contract_dims[0] = IndexPair<TensorIndex>(0, 0);
		}
		else {
			// row-major: output.patches.contract(kernel)
			contract_dims[0] = IndexPair<TensorIndex>(1, 1);
		}

		// Post contraction, the dimensions of the input_backprop is
		//  channels X input_rows X input_cols X OTHERS
		DSizes<TensorIndex, NumDims> post_contract_dims;
		if(isColMajor) {
			post_contract_dims[0] = kernelChannels;
			post_contract_dims[1] = inputRows;
			post_contract_dims[2] = inputCols;
			for(int i = 3; i < NumDims; ++i) {
				post_contract_dims[i] = out.dimension(i);
			}
		}
		else {
			post_contract_dims[NumDims - 1] = kernelChannels;
			post_contract_dims[NumDims - 2] = inputRows;
			post_contract_dims[NumDims - 3] = inputCols;
			for(int i = 0; i < NumDims - 3; ++i) {
				post_contract_dims[i] = out.dimension(i);
			}
		}

		return choose(
			Cond<internal::traits<OutputBackward>::Layout == ColMajor>(),
			kernel.reverse(kernel_reverse)
			.shuffle(kernel_shuffle)
			.reshape(kernel_dims)
			.eval()
			.contract(
			output_backward
			.extract_image_patches(
			kernelRows, kernelCols, 1, 1, row_in_stride,
			col_in_stride, row_stride, col_stride, padding_top,
			padding_bottom, padding_left, padding_right, OutScalar(0))
			.reshape(pre_contract_dims),
			contract_dims)
			.reshape(post_contract_dims),
			output_backward
			.extract_image_patches(kernelRows, kernelCols, 1, 1, row_in_stride,
			col_in_stride, row_stride, col_stride,
			padding_top, padding_bottom, padding_left,
			padding_right, OutScalar(0))
			.reshape(pre_contract_dims)
			.contract(kernel.reverse(kernel_reverse)
			.shuffle(kernel_shuffle)
			.reshape(kernel_dims)
			.eval(),
			contract_dims)
			.reshape(post_contract_dims));
	}

	template <typename OutputBackward, typename Input>
	EIGEN_ALWAYS_INLINE static const typename internal::conditional <
		internal::traits<OutputBackward>::Layout == ColMajor,
		TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index, 4>,
		const TensorContractionOp<
		const array<IndexPair<typename internal::traits<Input>::Index>, 1>,
		const TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index, 2>,
		const OutputBackward>,
		const TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index, 2>,
		const TensorImagePatchOp<Dynamic, Dynamic, const Input> > > >,
		TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index, 4>,
		const TensorContractionOp<
		const array<IndexPair<typename internal::traits<Input>::Index>, 1>,
		const TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index, 2>,
		const TensorImagePatchOp<Dynamic, Dynamic, const Input> >,
		const TensorReshapingOp<
		const DSizes<typename internal::traits<Input>::Index, 2>,
		const OutputBackward> > > > ::type
		SpatialConvolutionBackwardKernel(
			const Input& input, const OutputBackward& output_backward,
			typename internal::traits<Input>::Index kernelRows,
			typename internal::traits<Input>::Index kernelCols,
			const DenseIndex row_stride = 1, const DenseIndex col_stride = 1,
			const DenseIndex row_in_stride = 1, const DenseIndex col_in_stride = 1) {
		typedef typename internal::traits<Input>::Index TensorIndex;
		typedef typename internal::traits<OutputBackward>::Scalar OutScalar;
		TensorRef<Tensor<typename internal::traits<Input>::Scalar,
			internal::traits<Input>::NumDimensions,
			internal::traits<Input>::Layout, TensorIndex> >
			in(input);
		TensorRef<Tensor<OutScalar, internal::traits<OutputBackward>::NumDimensions,
			internal::traits<OutputBackward>::Layout, TensorIndex> >
			out(output_backward);

		EIGEN_STATIC_ASSERT(internal::traits<Input>::Layout ==
			internal::traits<OutputBackward>::Layout,
			YOU_MADE_A_PROGRAMMING_MISTAKE);

		// stride and in_stride cannot both be larger than 1
		eigen_assert(!(row_stride > 1 && row_in_stride > 1) &&
			!(col_stride > 1 && col_in_stride > 1));

		static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

		static const int NumDims = internal::traits<Input>::NumDimensions;
		EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions ==
			internal::traits<OutputBackward>::NumDimensions,
			YOU_MADE_A_PROGRAMMING_MISTAKE);

		const TensorIndex inputRows =
			isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
		const TensorIndex inputCols =
			isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);

		const TensorIndex outputRows = isColMajor
			? output_backward.dimension(1)
			: output_backward.dimension(NumDims - 2);
		const TensorIndex outputCols = isColMajor
			? output_backward.dimension(2)
			: output_backward.dimension(NumDims - 3);

		// Number of filters to apply. This is the same as the output depth of the
		// result
		const TensorIndex kernelFilters =
			isColMajor ? out.dimensions()[0] : out.dimensions()[NumDims - 1];

		// Number of channels. This is the same as the input depth.
		const TensorIndex kernelChannels =
			isColMajor ? in.dimensions()[0] : in.dimensions()[NumDims - 1];

		// This is the effective kernel size, taking into account the (*_in_stride -
		// 1) zero-values
		// inserted between consecutive kernel elements in atrous convolution
		const TensorIndex kernelRowsEff =
			kernelRows + (kernelRows - 1) * (row_in_stride - 1);
		const TensorIndex kernelColsEff =
			kernelCols + (kernelCols - 1) * (col_in_stride - 1);

		// Computing the forward padding
		const TensorIndex padRows = numext::maxi<Index>(
			0, (outputRows - 1) * row_stride + kernelRowsEff - inputRows);
		const TensorIndex padCols = numext::maxi<Index>(
			0, (outputCols - 1) * col_stride + kernelColsEff - inputCols);
		const TensorIndex padding_top = padRows / 2;
		const TensorIndex padding_bottom = padRows - padding_top;
		const TensorIndex padding_left = padCols / 2;
		const TensorIndex padding_right = padCols - padding_left;

		// Reshaped out
		DSizes<TensorIndex, 2> output_dims;
		if(isColMajor) {
			output_dims[0] = kernelFilters;
			output_dims[1] = outputRows * outputCols;
			for(int i = 3; i < NumDims; ++i) {
				output_dims[1] *= out.dimension(i);
			}
		}
		else {
			output_dims[1] = kernelFilters;
			output_dims[0] = outputCols * outputRows;
			for(int i = 0; i < NumDims - 3; ++i) {
				output_dims[0] *= out.dimension(i);
			}
		}

		// Reshaped extract_image_patches(in)
		DSizes<TensorIndex, 2> pre_contract_dims;
		if(isColMajor) {
			pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
			pre_contract_dims[1] = outputRows * outputCols;
			for(int i = 3; i < NumDims; ++i) {
				pre_contract_dims[1] *= in.dimension(i);
			}
			eigen_assert(output_dims[1] == pre_contract_dims[1]);
		}
		else {
			pre_contract_dims[1] = kernelCols * kernelRows * kernelChannels;
			pre_contract_dims[0] = outputRows * outputCols;
			for(int i = 0; i < NumDims - 3; ++i) {
				pre_contract_dims[0] *= in.dimension(i);
			}
			eigen_assert(output_dims[0] == pre_contract_dims[0]);
		}

		// We will contract along the collapsed dimension that contains the
		// outputCols, outputRows and OTHERS.
		array<IndexPair<TensorIndex>, 1> contract_dims;
		if(isColMajor) {
			// col-major: output_backward.contract(input.patches)
			contract_dims[0] = IndexPair<TensorIndex>(1, 1);
		}
		else {
			// row-major: input.patches.contract(output_backward)
			contract_dims[0] = IndexPair<TensorIndex>(0, 0);
		}

		// After the contraction, the kernel will have the desired shape
		// out_depth X in_shape X kernel_rows X kernel_cols
		DSizes<TensorIndex, 4> kernel_dims;
		if(isColMajor) {
			kernel_dims[0] = kernelFilters;
			kernel_dims[1] = kernelChannels;
			kernel_dims[2] = kernelRows;
			kernel_dims[3] = kernelCols;
		}
		else {
			kernel_dims[3] = kernelFilters;
			kernel_dims[2] = kernelChannels;
			kernel_dims[1] = kernelRows;
			kernel_dims[0] = kernelCols;
		}

		return choose(
			Cond<internal::traits<Input>::Layout == ColMajor>(),
			output_backward.reshape(output_dims)
			.contract(
			input
			.extract_image_patches(
			kernelRows, kernelCols, row_stride, col_stride,
			row_in_stride, col_in_stride, 1, 1, padding_top,
			padding_bottom, padding_left, padding_right, OutScalar(0))
			.reshape(pre_contract_dims),
			contract_dims)
			.reshape(kernel_dims),
			input
			.extract_image_patches(kernelRows, kernelCols, row_stride, col_stride,
			row_in_stride, col_in_stride, 1, 1,
			padding_top, padding_bottom, padding_left,
			padding_right, OutScalar(0))
			.reshape(pre_contract_dims)
			.contract(output_backward.reshape(output_dims), contract_dims)
			.reshape(kernel_dims));
	}
}