#include "nns/nns.h"

static constexpr int seg_lens_50[] = {3,4,6,3};
static constexpr int seg_lens_101[] = {3,4,23,3};
static constexpr int seg_lens_152[] = {3,8,36,3};

// Bottleneck block.
static void btn_blk(
	Network& n, Network::lid_t& prev,
	int seg, int blk, bool has_br,
	len_t in_C, len_t mid_C, len_t out_C,
	len_t H_len, len_t strd=1
){
	std::string name = "conv_"+std::to_string(seg)+"_"+std::to_string(blk)+"_";
	n.add(NLAYER(name+"a", Conv, C=in_C, K=mid_C, H=H_len, sH=strd), {prev});
	n.add(NLAYER(name+"b", Conv, C=mid_C, K=mid_C, H=H_len, R=3));
	auto layer_c = n.add(NLAYER(name+"c", Conv, C=mid_C, K=out_C, H=H_len));
	if(has_br){
		prev = n.add(NLAYER("conv_"+std::to_string(seg)+"_br", Conv,
				C=in_C, K=out_C, H=H_len, sH=strd), {prev});
	}
	prev = n.add(NLAYER(name+"res", Eltwise, K=out_C, H=H_len, N=2), {prev, layer_c});
}

static void group_btn_blk(
	Network& n, Network::lid_t& prev,
	int seg, int blk, bool has_br,
	len_t in_C, len_t mid_C, len_t out_C,
	len_t H_len, len_t strd=1
){
	std::string name = "conv_"+std::to_string(seg)+"_"+std::to_string(blk)+"_";
	n.add(NLAYER(name+"a", Conv, C=in_C, K=mid_C, H=H_len, sH=strd), {prev});
	n.add(NLAYER(name+"b", GroupConv, G=32, C=mid_C, K=mid_C, H=H_len, R=3));
	auto layer_c = n.add(NLAYER(name+"c", Conv, C=mid_C, K=out_C, H=H_len));
	if(has_br){
		prev = n.add(NLAYER("conv_"+std::to_string(seg)+"_br", Conv,
				C=in_C, K=out_C, H=H_len, sH=strd), {prev});
	}
	prev = n.add(NLAYER(name+"res", Eltwise, K=out_C, H=H_len, N=2), {prev, layer_c});
}

static Network gen_resnet(const int* seg_lens){
	len_t H_lens[] = {56,28,14,7};
	len_t mid_Cs[] = {64,128,256,512};
	len_t out_Cs[] = {256,512,1024,2048};
	len_t init_C = 64;

	Network::lid_t prev;
	Network n;
	InputData input("input", fmap_shape(3,224));

	n.add(NLAYER("conv1", Conv, C=3, K=init_C, H=112, R=7, sH=2), {}, 0, {input});
	prev = n.add(NLAYER("pool1", Pooling, K=init_C, H=56, R=3, sH=2));

	len_t strd = 1;
	len_t in_C = init_C;
	len_t H_len, mid_C, out_C;
	for(int seg=2; seg<6; ++seg){
		H_len = H_lens[seg-2];
		mid_C = mid_Cs[seg-2];
		out_C = out_Cs[seg-2];

		for(int blk=0; blk<seg_lens[seg-2]; ++blk){
			btn_blk(n, prev, seg, blk, blk==0,
					in_C, mid_C, out_C, H_len, strd);
			strd = 1;
			in_C = out_C;
		}
		strd = 2;
	}
	n.add(NLAYER("pool5", Pooling, K=in_C, H=1, R=H_len), {prev});
	n.add(NLAYER("fc", FC, C=in_C, K=1000));
	return n;
}

static Network gen_resnext(const int* seg_lens){
	len_t H_lens[] = {56,28,14,7};
	len_t mid_Cs[] = {64,128,256,512};
	len_t out_Cs[] = {256,512,1024,2048};
	len_t init_C = 64;

	Network::lid_t prev;
	Network n;
	InputData input("input", fmap_shape(3,224));

	n.add(NLAYER("conv1", Conv, C=3, K=init_C, H=112, R=7, sH=2), {}, 0, {input});
	prev = n.add(NLAYER("pool1", Pooling, K=init_C, H=56, R=3, sH=2));

	len_t strd = 1;
	len_t in_C = init_C;
	len_t H_len, mid_C, out_C;
	for(int seg=2; seg<6; ++seg){
		H_len = H_lens[seg-2];
		mid_C = mid_Cs[seg-2];
		out_C = out_Cs[seg-2];

		for(int blk=0; blk<seg_lens[seg-2]; ++blk){
			group_btn_blk(n, prev, seg, blk, blk==0,
					in_C, mid_C, out_C, H_len, strd);
			strd = 1;
			in_C = out_C;
		}
		strd = 2;
	}
	n.add(NLAYER("pool5", Pooling, K=in_C, H=1, R=H_len), {prev});
	n.add(NLAYER("fc", FC, C=in_C, K=1000));
	return n;
}
const Network resnet50 = gen_resnet(seg_lens_50);
const Network resnet101 = gen_resnet(seg_lens_101);
const Network resnet152 = gen_resnet(seg_lens_152);
const Network resnext50 = gen_resnext(seg_lens_50);
