#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
//#include <mlpack/methods/ann/layer/identity_output_layer.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/Logging.hpp"

#include <iostream>

using namespace mlpack;


class TwoLayerNetwork {
	public:

	std::tuple <	ann::Linear<>,
					ann::BaseLayer<ann::LogisticFunction>,
					ann::Linear<>,
					ann::BaseLayer<ann::IdentityFunction>
				> modules;

	ann::FFN < 	ann::IdentityLayer<>,	   
				ann::RandomInitialization
				> net;
	
	TwoLayerNetwork(int inputVectorSize, const std::vector<int> &hiddenLayerSize, int outputVectorSize){
				modules= std::tuple<ann::Linear<>,
					ann::BaseLayer<ann::LogisticFunction>,
					ann::Linear<>,
					ann::BaseLayer<ann::IdentityFunction>
				>
				(ann::Linear<>(inputVectorSize,hiddenLayerSize[0]),
					ann::BaseLayer<ann::LogisticFunction>(),
					ann::Linear<>(hiddenLayerSize[0], outputVectorSize),
					ann::BaseLayer<ann::IdentityFunction>()
				   );
				//ann::FFN<ann::IdentityLayer<>,ann::RandomInitialization> net = ann::FFN<ann::IdentityLayer<>,ann::RandomInitialization>(ann::IdentityLayer<>(),ann::RandomInitialization());
				ann::FFN<ann::MeanSquaredError<>,ann::RandomInitialization> net;
				net.Add<ann::Linear<>>(inputVectorSize,hiddenLayerSize[0]);
				net.Add<ann::BaseLayer<ann::LogisticFunction>>();
				net.Add<ann::Linear<>>(hiddenLayerSize[0], outputVectorSize);
				net.Add<ann::BaseLayer<ann::IdentityFunction>>();
			}
			 //constructor

	private:
};
