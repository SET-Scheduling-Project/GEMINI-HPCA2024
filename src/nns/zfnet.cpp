#include "nns/nns.h"

const Network zfnet = []{
	Network n;
	InputData input("input", fmap_shape(3, 224));
	n.add(NLAYER("conv1", Conv, C=3, K=96, H=110, R=7, sH=2), {}, 0, {input});
    n.add(NLAYER("pool1", Pooling, K=96, H=55, R=3, sH=2));
    n.add(NLAYER("conv2", Conv, C=96, K=256, H=26, R=5, sH=2));
    n.add(NLAYER("pool2", Pooling, K=256, H=13, R=3, sH=2));
    n.add(NLAYER("conv3", Conv, C=256, K=512, H=13, R=3));
    n.add(NLAYER("conv4", Conv, C=512, K=1024, H=13, R=3));
    n.add(NLAYER("conv5", Conv, C=1024, K=512, H=13, R=3));
    n.add(NLAYER("pool3", Pooling, K=512, H=6, R=3, sH=2));
    n.add(NLAYER("fc1", FC, C=512, K=4096, IH=6));
    n.add(NLAYER("fc2", FC, C=4096, K=4096));
    n.add(NLAYER("fc3", FC, C=4096, K=1000));
	return n;
}();
