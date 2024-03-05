#include "nns/nns.h"

static Network::layer_set add_inception(
		Network& n, std::string incp_id,
		len_t sfmap, len_t nfmaps_in, len_t nfmaps_1, len_t nfmaps_3r,
		len_t nfmaps_3, len_t nfmaps_5r, len_t nfmaps_5, len_t nfmaps_pool,
		const Network::layer_set& prevs){

	Network::lid_t a,b,c,d;
	//Add an inception module to the network.
	std::string pfx = "inception_" + incp_id + "_";
	// 1x1 branch.
	a=n.add(NLAYER(pfx + "1x1", Conv, C=nfmaps_in, K=nfmaps_1, H=sfmap), prevs);
	// 3x3 branch.
	n.add(NLAYER(pfx + "3x3_reduce", Conv, C=nfmaps_in, K=nfmaps_3r, H=sfmap), prevs);
	b=n.add(NLAYER(pfx + "3x3", Conv, C=nfmaps_3r, K=nfmaps_3, H=sfmap, R=3));
	// 5x5 branch.
	n.add(NLAYER(pfx + "5x5_reduce", Conv, C=nfmaps_in, K=nfmaps_5r, H=sfmap, R=1),prevs);
	c=n.add(NLAYER(pfx + "5x5", Conv, C=nfmaps_5r, K=nfmaps_5, H=sfmap, R=5));
	// Pooling branch.
	n.add(NLAYER(pfx + "pool", Pooling, K = nfmaps_in, H = sfmap, R=3, sH=1), prevs);
	d=n.add(NLAYER(pfx + "pool_proj", Conv, C=nfmaps_in, K=nfmaps_pool, H=sfmap, R=1));
	// Merge branches.
	return {a,b,c,d};
}


const Network googlenet = []{
	Network::lid_t prev;
	Network::layer_set prevs;
	Network n;
	InputData input("input", fmap_shape(3,224));

	n.add(NLAYER("conv1", Conv, C=3, K=64, H=112, R=7, sH=2), {}, 0, {input});
	n.add(NLAYER("pool1", Pooling, K=64, H=56, R=3, sH=2));
	n.add(NLAYER("conv2_3x3_reduce", Conv, C=64, K=64, H=56, R=1));
	n.add(NLAYER("conv2_3x3", Conv, C=64, K=192, H=56, R=3));
	prev = n.add(NLAYER("pool2", Pooling, K=192, H=28, R=3, sH=2));

	prevs = {prev};
	prevs = add_inception(n, "3a", 28, 192, 64, 96, 128, 16, 32, 32, prevs);
	prevs = add_inception(n, "3b", 28, 256, 128, 128, 192, 32, 96, 64, prevs);

	prev = n.add(NLAYER("pool3", Pooling, K=480, H=14, R=3, sH=2), prevs);
	prevs = {prev};

	// Inception 4.
	prevs = add_inception(n, "4a", 14, 480, 192, 96, 208, 16, 48, 64, prevs);
	prevs = add_inception(n, "4b", 14, 512, 160, 112, 224, 24, 64, 64, prevs);
	prevs = add_inception(n, "4c", 14, 512, 128, 128, 256, 24, 64, 64, prevs);
	prevs = add_inception(n, "4d", 14, 512, 112, 144, 288, 32, 64, 64, prevs);
	prevs = add_inception(n, "4e", 14, 528, 256, 160, 320, 32, 128, 128, prevs);

	prev=n.add(NLAYER("pool4", Pooling, K=832, H=7, R=3, sH=2), prevs);
	prevs = {prev};

	// Inception 5.
	prevs = add_inception(n, "5a", 7, 832, 256, 160, 320, 32, 128, 128, prevs);
	prevs = add_inception(n, "5b", 7, 832, 384, 192, 384, 48, 128, 128, prevs);

	n.add(NLAYER("pool5", Pooling, K=1024, H=1, R=7), prevs);
	n.add(NLAYER("fc", FC, C=1024, K=1000));
	return n;
}();

