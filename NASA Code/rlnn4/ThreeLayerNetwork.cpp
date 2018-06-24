#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/identity_output_layer.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/Logging.hpp"

#include <iostream>

using namespace mlpack;


class ThreeLayerNetwork {
	public:

	std::tuple <	ann::LinearLayer<>,
					ann::BaseLayer<ann::LogisticFunction>,
					ann::LinearLayer<>,
					ann::BaseLayer<ann::LogisticFunction>,
					ann::LinearLayer<>,
					ann::BaseLayer<ann::IdentityFunction>
				> modules;

	ann::FFN < 	decltype(modules),
				ann::IdentityOutputLayer,	   
				ann::RandomInitialization,
				ann::MeanSquaredErrorFunction
			> net;
	
	ThreeLayerNetwork(int inputVectorSize, const std::vector<int> &hiddenLayerSize, int outputVectorSize)
			: 	
				modules(ann::LinearLayer<>(inputVectorSize,hiddenLayerSize[0]),
					ann::BaseLayer<ann::LogisticFunction>(),
					ann::LinearLayer<>(hiddenLayerSize[0], hiddenLayerSize[1]),
					ann::BaseLayer<ann::LogisticFunction>(),
					ann::LinearLayer<>(hiddenLayerSize[1], outputVectorSize),
					ann::BaseLayer<ann::IdentityFunction>()
				   ),
				net(modules,ann::IdentityOutputLayer(),ann::RandomInitialization(),ann::MeanSquaredErrorFunction())
			{} //constructor

	private:
};
