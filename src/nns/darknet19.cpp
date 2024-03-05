#include "nns/nns.h"

const Network darknet19 = []{
	Network n;
	InputData input("input", fmap_shape(3, 224));
	n.add(NLAYER("conv1", Conv, C=3, K=32, H=224, R=3), {}, 0, {input});
	n.add(NLAYER("pool1", Pooling, K=32, H=112, R=2));
	n.add(NLAYER("conv2", Conv, C=32, K=64, H=112, R=3));
	n.add(NLAYER("pool2", Pooling, K=64, H=56, R=2));

	n.add(NLAYER("conv3", Conv, C=64, K=128, H=56, R=3));
	n.add(NLAYER("conv4", Conv, C=128, K=64, H=56, R=1));
	n.add(NLAYER("conv5", Conv, C=64, K=128, H=56, R=3));
	n.add(NLAYER("pool3", Pooling, K=128, H=28, R=2));

	n.add(NLAYER("conv6", Conv, C=128, K=256, H=28, R=3));
	n.add(NLAYER("conv7", Conv, C=256, K=128, H=28, R=1));
	n.add(NLAYER("conv8", Conv, C=128, K=256, H=28, R=3));
	n.add(NLAYER("pool4", Pooling, K=256, H=14, R=2));

	n.add(NLAYER("conv9", Conv, C=256, K=512, H=14, R=3));
	n.add(NLAYER("conv10", Conv, C=512, K=256, H=14, R=1));
	n.add(NLAYER("conv11", Conv, C=256, K=512, H=14, R=3));
	n.add(NLAYER("conv12", Conv, C=512, K=256, H=14, R=1));
	n.add(NLAYER("conv13", Conv, C=256, K=512, H=14, R=3));
	n.add(NLAYER("pool5", Pooling, K=512, H=7, R=2));

	n.add(NLAYER("conv14", Conv, C=512, K=1024, H=7, R=3));
	n.add(NLAYER("conv15", Conv, C=1024, K=512, H=7, R=1));
	n.add(NLAYER("conv16", Conv, C=512, K=1024, H=7, R=3));
	n.add(NLAYER("conv17", Conv, C=1024, K=512, H=7, R=1));
	n.add(NLAYER("conv18", Conv, C=512, K=1024, H=7, R=3));

	n.add(NLAYER("conv19", Conv, C=1024, K=1000, H=7, R=1));

	n.add(NLAYER("pool_avg", Pooling, K=1000, H=1, R=7));
	return n;
}();
