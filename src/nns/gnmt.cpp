#include "nns/nns.h"

// GNMT Block.
static Network::lid_t gnmt_blk(
	const std::string& name, Network& n, Network::lid_t xin, len_t len
){
	InputData Cin(name+"_cin", fmap_shape(len, 1));
	InputData Hin(name+"_hin", fmap_shape(len, 1));
	Network::lid_t f_id, i_id, cand_id, o_id, fout_id, iout_id, cout_id, hout_id;

	f_id = n.add(NLAYER(name+"_f", FC, C=2*len, K=len), {xin}, 0, {Hin});
	i_id = n.add(NLAYER(name+"_i", FC, C=2*len, K=len), {xin}, 0, {Hin});
	cand_id = n.add(NLAYER(name+"_cand", FC, C=2*len, K=len), {xin}, 0, {Hin});
	o_id = n.add(NLAYER(name+"_o", FC, C=2*len, K=len), {xin}, 0, {Hin});

	fout_id = n.add(NLAYER(name+"_fout", Eltwise, K=len, H=1, N=2), {f_id}, 0, {Cin});
	iout_id = n.add(NLAYER(name+"_iout", Eltwise, K=len, H=1, N=2), {i_id, cand_id});
	cout_id = n.add(NLAYER(name+"_cout", Eltwise, K=len, H=1, N=2), {fout_id, iout_id});
	cout_id = n.add(NLAYER(name+"_tanh", PTP, K=len, H=1), {cout_id});
	hout_id = n.add(NLAYER(name+"_hout", Eltwise, K=len, H=1, N=2), {cout_id, o_id});
	return hout_id;
}

static Network gen_lstm(int nblocks, bool has_res, int len = 1000){
	Network n;
	InputData input("input", fmap_shape(len, 1));
	Network::lid_t xin, next_xin;

	xin = n.add(NLAYER("word_embed", PTP, K=len, H=1), {}, 0, {input});

	for(int block_id=1; block_id<=nblocks; ++block_id){
		std::string block_name = "block" + std::to_string(block_id);
		next_xin = gnmt_blk(block_name, n, xin, len);
		if(has_res){
			next_xin = n.add(NLAYER(block_name + "_add", Eltwise, K=len, H=1, N=2), {xin, next_xin});
		}
		xin = next_xin;
	}
	n.add(NLAYER("Wd", PTP, K=len, H=1), {xin});
	return n;
}

const Network gnmt = gen_lstm(8, true);
const Network lstm = gen_lstm(8, false);
