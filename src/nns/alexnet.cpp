#include "nns/nns.h"

const Network alexnet = []{
	Network n;
	InputData input("input", fmap_shape(3, 224));
	auto conv1a = n.add(NLAYER("conv1_a", Conv, C=3, K=48, H=55, R=11, sH=4), {}, 0, {input});
	auto conv1b = n.add(NLAYER("conv1_b", Conv, C=3, K=48, H=55, R=11, sH=4), {}, 0, {input});
	auto pool1a = n.add(NLAYER("pool1_a", Pooling, K=48, H=27, R=3, sH=2), {conv1a});
	auto pool1b = n.add(NLAYER("pool1_b", Pooling, K=48, H=27, R=3, sH=2), {conv1b});

	auto conv2a = n.add(NLAYER("conv2_a", Conv, C=48, K=128, H=27, R=5), {pool1a});
	auto conv2b = n.add(NLAYER("conv2_b", Conv, C=48, K=128, H=27, R=5), {pool1b});
	auto pool2a = n.add(NLAYER("pool2_a", Pooling, K=128, H=13, R=3, sH=2), {conv2a});
	auto pool2b = n.add(NLAYER("pool2_b", Pooling, K=128, H=13, R=3, sH=2), {conv2b});

	auto conv3a = n.add(NLAYER("conv3_a", Conv, C=256, K=192, H=13, R=3), {pool2a,pool2b});
	auto conv3b = n.add(NLAYER("conv3_b", Conv, C=256, K=192, H=13, R=3), {pool2a,pool2b});
	auto conv4a = n.add(NLAYER("conv4_a", Conv, C=192, K=192, H=13, R=3), {conv3a});
	auto conv4b = n.add(NLAYER("conv4_b", Conv, C=192, K=192, H=13, R=3), {conv3b});
	auto conv5a = n.add(NLAYER("conv5_a", Conv, C=192, K=128, H=13, R=3), {conv4a});
	auto conv5b = n.add(NLAYER("conv5_b", Conv, C=192, K=128, H=13, R=3), {conv4b});
	auto pool3a = n.add(NLAYER("pool3_a", Pooling, K=128, H=6, R=3, sH=2), {conv5a});
	auto pool3b = n.add(NLAYER("pool3_b", Pooling, K=128, H=6, R=3, sH=2), {conv5b});

	n.add(NLAYER("fc1", FC, C=256, K=4096, IH=6), {pool3a, pool3b});
	n.add(NLAYER("fc2", FC, C=4096, K=4096));
	n.add(NLAYER("fc3", FC, C=4096, K=1000));
	return n;
}();
