#include "nns/nns.h"
#include "util.h"
#include <cassert>


typedef TransposeLayer::dim Ldims;

static lid_t add_attention(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize, len_t decode_len,
		lid_t prev
){

	lid_t Q, K, V, QK, QK_elt, QKV;
	Network::layer_set Ks;
	Q = n.add(NLAYER(name + "_Q", Conv, H=len, W=1, C=numG*gSize), {prev});

	len_t kv_len;
	if(decode_len == 0){
		kv_len = len;
		for(len_t i=0; i<numG; ++i){
			n.add(NLAYER(name + "_K" + std::to_string(i), Conv, C=numG*gSize, K=gSize, H=len, W=1), {prev});
			K = n.add(NLAYER(name + "_Kt" + std::to_string(i), Transpose, K=len, H=gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C));
			Ks.push_back(K);
		}
		// K = n.add(NLAYER(name + "_K", PTP, K=numG*len, H=gSize, W=1), Ks);
		V = n.add(NLAYER(name + "_V", Conv, C=numG*gSize, H=len, W=1), {prev});
	}else{
		assert(len == 1);
		kv_len = len + decode_len;

		InputData extK(name + "_Kext", fmap_shape(numG*decode_len, gSize, 1));
		InputData extV(name + "_Vext", fmap_shape(decode_len, numG*gSize, 1));

		for(len_t i=0; i<numG; ++i){
			n.add(NLAYER(name + "_K" + std::to_string(i), Conv, C=numG*gSize, K=gSize, H=len, W=1), {prev});
			K = n.add(NLAYER(name + "_Kt" + std::to_string(i), Transpose, K=len, H=gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C));
			Ks.push_back(K);
		}
		K = n.add(NLAYER(name + "_K", PTP, K=numG*kv_len, H=gSize, W=1), Ks, 0, {extK});
		Ks = {K};
		V = n.add(NLAYER(name + "_V", Conv, C=numG*gSize, H=len, W=1), {prev});
		V = n.add(NLAYER(name + "_Vt1", Transpose, K=len, H=numG*gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {V});
		V = n.add(NLAYER(name + "_Vt2", Transpose, K=numG*gSize, H=kv_len, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {V}, 0, {extV});
	}
	QK = n.add(NLAYER(name + "_QK", GroupConv, H=len, W=1, C=numG*gSize, K = numG*kv_len, G=numG), {Q}, 0, {}, Ks);
	QK_elt = n.add(NLAYER(name + "_QK_elt", PTP, K=numG*kv_len, H=len, W=1), {QK});
	QKV = n.add(NLAYER(name + "_QKV", GroupConv, H=len, W=1, C=numG*kv_len, K=numG*gSize, G=numG), {QK_elt}, 0, {}, {V});
	return n.add(NLAYER(name + "_FC", Conv, H=len, W=1, C=numG*gSize), {QKV});
}

static lid_t add_trans_block(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize, len_t ff_len, len_t decode_len,
		lid_t prev
){

	lid_t next_prev;
	next_prev = add_attention(n, name, len, numG, gSize, decode_len, prev);
	prev = n.add(NLAYER(name + "_elt1", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
	n.add(NLAYER(name + "_feedfwd1", Conv, C=numG*gSize, K=ff_len, H=len, W=1));
	next_prev = n.add(NLAYER(name + "_feedfwd2", Conv, C=ff_len, K=numG*gSize, H=len, W=1));
	return n.add(NLAYER(name + "_elt2", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
}

static Network create_transformer(
		len_t numG, len_t gSize, len_t nBlock, bool is_prefill,
		len_t vocab_len = 1000, len_t len = 512, len_t ff_len = 0
){
	// Default settings.
	if(ff_len == 0){
		ff_len = 4 * len;
	}

	len_t decode_len = 0;
	if(!is_prefill){
		decode_len = len;
		len = 1;
	}

	// Length of embedding
	len_t totG = numG * gSize;
	// Number of embedding
	len_t curH = len;

	lid_t block_prev;
	Network::layer_set prevs;
	Network n;

	InputData input_layer("input_layer", fmap_shape(totG, curH, 1));
	block_prev = n.add(NLAYER("word_embed", PTP, K=totG, H=curH, W=1), {}, 0, {input_layer});
	for(len_t i=1; i<=nBlock; ++i){
		block_prev = add_trans_block(n, "block"+std::to_string(i), len, numG, gSize, ff_len, decode_len, block_prev);
	}
	n.add(NLAYER("proj", Conv, C=totG, K=vocab_len, H=curH, W=1), {block_prev});
	return n;
};

/*
 * Since the complete networks have too many layers,
 * and all blocks in the network are identical,
 * a single block is provided for each network.
 *
 * To run the full network, one can uncomment the networks below.
 * Also remember to uncomment in "nns/nns.h", add the network in "main.cpp",
 * and increase MAX_BITS_IN_BS in "bitset.h".
 */

/*
 * BERT-Large
 *
 * numG       = 16
 * gSize      = 64
 * nBlock     = 24
 * is_prefill = true
*/
// const Network BERT = create_transformer(16, 64, 24, true);
const Network BERT_block = create_transformer(16, 64, 1, true);

/*
 * GPT2-XL at Prefill stage
 *
 * numG       = 25
 * gSize      = 64
 * nBlock     = 48
 * is_prefill = true
*/
// const Network GPT2_prefill = create_transformer(25, 64, 48, true);
const Network GPT2_prefill_block = create_transformer(25, 64, 1, true);

/*
 * GPT2-XL at Decode stage
 *
 * numG       = 25
 * gSize      = 64
 * nBlock     = 48
 * is_prefill = false
*/
// const Network GPT2_decode = create_transformer(25, 64, 48, false);
const Network GPT2_decode_block = create_transformer(25, 64, 1, false);