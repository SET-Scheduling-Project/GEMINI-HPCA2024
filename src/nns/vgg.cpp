#include "nns/nns.h"

static constexpr len_t param_19[][4]=
{{3, 64, 224, 3},
{64, 64, 224, 3},
{64, 128, 112, 3},
{128, 128, 112, 3},

{128, 256, 56, 3},
{256, 256, 56, 3},
{256, 256, 56, 3},
{256, 256, 56, 3},

{256, 512, 28, 3},
{512, 512, 28, 3},
{512, 512, 28, 3},
{512, 512, 28, 3},

{512, 512, 14, 3},
{512, 512, 14, 3},
{512, 512, 14, 3},
{512, 512, 14, 3},
// FC
{512, 4096, 1, 7},
{4096, 4096, 1, 1},
{4096, 1000, 1, 1}};

const Network vgg19 = []{
	Network n;
	InputData input("input", fmap_shape(3,224));
	const len_t* cur_size = param_19[0];
	n.add(NLAYER("conv1", Conv, C=cur_size[0], K=cur_size[1], H=cur_size[2], R=cur_size[3]), {}, 0, {input});
	int npool = 0;
	for(int i=2;i<=16;++i){
		if(cur_size[2] != param_19[i-1][2]){
			n.add(NLAYER("pool" + std::to_string(++npool), Pooling, K=cur_size[1], H=param_19[i-1][2], R=2));
		}
		cur_size=param_19[i-1];
		n.add(NLAYER("conv"+std::to_string(i), Conv, C=cur_size[0], K=cur_size[1], H=cur_size[2], R=cur_size[3]));
	}
	n.add(NLAYER("pool" + std::to_string(++npool), Pooling, K=cur_size[1], H=param_19[16][3], R=2));
	//n.add(NLAYER("fc1", FC, C=param_19[16][0], K=param_19[16][1], IH = param_19[16][3]));
	//n.add(NLAYER("fc2", FC, C=param_19[17][0], K=param_19[17][1]));
	//n.add(NLAYER("fc3", FC, C=param_19[18][0], K=param_19[18][1]));
	return n;
}();
