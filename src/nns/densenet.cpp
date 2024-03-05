#include "nns/nns.h"
#include <string>

static int add_dblock(Network& network, int incp_id, int& sfmap, int num_layers, int nfmaps_in, Network::layer_set& prevs, int k=32, int bn_size = 4)
{
    std::string pfx = std::string("block_") + std::to_string(incp_id) + "_";
    int fmap_bn = bn_size * k;

	for(int i=0;i<num_layers;++i){
        int fmap_in = nfmaps_in + i*k;
        std::string layer_1x1 = pfx + std::to_string(i) + "_1x1";
        std::string layer_3x3 = pfx + std::to_string(i) + "_3x3";
        network.add(NLAYER(layer_1x1.c_str(), Conv, C=fmap_in, K=fmap_bn, H=sfmap, R=1), prevs);
        auto layer3x3 = network.add(NLAYER(layer_3x3.c_str(), Conv, C=fmap_bn, K=k, H=sfmap, R=3));
		prevs.push_back(layer3x3);
    }
    int fmap_in = nfmaps_in + num_layers * k;
    return fmap_in;
}

static Network::lid_t add_trans(Network& network, int trans_id, int& fmap_in, int& sfmap, Network::layer_set prevs)
{
    std::string pfx = std::string("trans_") + std::to_string(trans_id) + "_";
    network.add(NLAYER((pfx + "1x1").c_str(), Conv, C=fmap_in, K=fmap_in/2, H=sfmap, R=1), prevs);
    auto ret = network.add(NLAYER((pfx + "pool").c_str(), Pooling, K=fmap_in/2, H=sfmap/2, R=2));
    fmap_in/=2;
    sfmap/=2;
    return ret;
}

const Network densenet = []{
	Network n;
	InputData input("input", fmap_shape(3, 224));
    int init_f = 64;
    int layer_num_list[4] = {6,12,24,16};
	n.add(NLAYER("conv0", Conv, C=3, K=init_f, H=112, R=7, sH=2), {}, 0, {input});
    auto pool0 = n.add(NLAYER("pool0", Pooling, K=init_f, H=56, R=3, sH=2));

    int sfmap = 56;
    int fmap_in = init_f;
	Network::layer_set prevs = {pool0};

    for(int i=0;i<4;++i)
    {
        int nlayers=layer_num_list[i];
        fmap_in = add_dblock(n, i+1, sfmap, nlayers, fmap_in, prevs);
        if(i!=3)
        {
            prevs = {add_trans(n, i+1, fmap_in, sfmap, prevs)};
        }
    }
    n.add(NLAYER("pool_avg", Pooling, K=fmap_in, H=1, R=sfmap), prevs);
    n.add(NLAYER("fc", FC, C=fmap_in, K=1000));

	return n;
}();
