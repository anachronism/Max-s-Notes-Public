#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
//#include <mlpack/methods/ann/layer/identity_layer.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/Logging.hpp"

#include <iostream>

using namespace mlpack;


class ThreeLayerNetwork {
	public:

	std::tuple <	ann::Linear<>,
					ann::BaseLayer<ann::LogisticFunction>,
					ann::Linear<>,
					ann::BaseLayer<ann::LogisticFunction>,
					ann::Linear<>,
					ann::BaseLayer<ann::IdentityFunction>
				> modules;

	// ann::FFN < 	decltype(modules),
	// 			ann::IdentityLayer<>,	   
	// 			ann::RandomInitialization,
	// 			ann::MeanSquaredError
	// 		> net;
	
	ann::FFN < 	//decltype(modules),
				ann::IdentityLayer<>,	   
				ann::RandomInitialization
			> net;

	ThreeLayerNetwork(int inputVectorSize, const std::vector<int> &hiddenLayerSize, int outputVectorSize){ 	
				modules = std::tuple<ann::Linear<>,
					ann::BaseLayer<ann::LogisticFunction>,
					ann::Linear<>,
					ann::BaseLayer<ann::LogisticFunction>,
					ann::Linear<>,
					ann::BaseLayer<ann::IdentityFunction>
				>(ann::Linear<>(inputVectorSize,hiddenLayerSize[0]),
					ann::BaseLayer<ann::LogisticFunction>(),
					ann::Linear<>(hiddenLayerSize[0], hiddenLayerSize[1]),
					ann::BaseLayer<ann::LogisticFunction>(),
					ann::Linear<>(hiddenLayerSize[1], outputVectorSize),
					ann::BaseLayer<ann::IdentityFunction>()
				   );
		//		net(modules,ann::IdentityLayer<>(),ann::RandomInitialization(),ann::MeanSquaredError<>())
		//ann::FFN<ann::IdentityLayer<>,ann::RandomInitialization> net = ann::FFN<ann::IdentityLayer<>,ann::RandomInitialization>(ann::IdentityLayer<>(),ann::RandomInitialization());
		ann::FFN<ann::MeanSquaredError<>,ann::RandomInitialization> net;
		net.Add<ann::Linear<>>(inputVectorSize,hiddenLayerSize[0]);
		net.Add<ann::BaseLayer<ann::LogisticFunction>>();
		net.Add<ann::Linear<>>(hiddenLayerSize[0], hiddenLayerSize[1]);
		net.Add<ann::BaseLayer<ann::LogisticFunction>>();
		net.Add<ann::Linear<>>(hiddenLayerSize[1], outputVectorSize);
		net.Add<ann::BaseLayer<ann::IdentityFunction>>();
		}

	private:
};
