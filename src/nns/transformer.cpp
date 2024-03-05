#include "nns/nns.h"

typedef TransposeLayer::dim Ldims;

static Network::lid_t add_attention(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize,
		Network::lid_t prevQ, Network::lid_t prevK, Network::lid_t prevV){


	Network::lid_t Q, K, V, QK, QK_elt, QKV;
	Network::layer_set Ks;
	Q = n.add(NLAYER(name + "_Q", Conv, H=len, W=1, C=numG*gSize), {prevQ});
	for(len_t i=0; i<numG; ++i){
		n.add(NLAYER(name + "_K" + std::to_string(i), Conv, C=numG*gSize, K=gSize, H=len, W=1), {prevK});
		K = n.add(NLAYER(name + "_Kt" + std::to_string(i), Transpose, K=len, H=gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C));
		Ks.push_back(K);
	}
	K = n.add(NLAYER(name + "_K", PTP, K=numG*len, H=gSize, W=1), Ks);
	V = n.add(NLAYER(name + "_V", Conv, C=numG*gSize, H=len, W=1), {prevV});
	QK = n.add(NLAYER(name + "_QK", GroupConv, H=len, W=1, C=numG*gSize, K = numG*len, G=numG), {Q}, 0, {}, {K});
	QK_elt = n.add(NLAYER(name + "_QK_elt", PTP, K=numG*len, H=len, W=1), {QK});
	QKV = n.add(NLAYER(name + "_QKV", GroupConv, H=len, W=1, C=numG*len, K=numG*gSize, G=numG), {QK_elt}, 0, {}, {V});
	return n.add(NLAYER(name + "_FC", Conv, H=len, W=1, C=numG*gSize), {QKV});
}

static Network::lid_t add_encoder(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize, len_t ff_len,
		Network::lid_t prev){
	Network::lid_t next_prev;
	next_prev = add_attention(n, name, len, numG, gSize, prev, prev, prev);
	prev = n.add(NLAYER(name + "_elt1", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
	n.add(NLAYER(name + "_feedfwd1", Conv, C=numG*gSize, K=ff_len, H=len, W=1));
	next_prev = n.add(NLAYER(name + "_feedfwd2", Conv, C=ff_len, K=numG*gSize, H=len, W=1));
	return n.add(NLAYER(name + "_elt2", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
}

static Network::lid_t add_decoder(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize, len_t ff_len,
		Network::lid_t prev, Network::lid_t enc_prev){
	Network::lid_t next_prev;
	next_prev = add_attention(n, name+"_1", len, numG, gSize, prev, prev, prev);
	prev = n.add(NLAYER(name + "_elt1", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
	next_prev = add_attention(n, name+"_2", len, numG, gSize, prev, enc_prev, enc_prev);
	prev = n.add(NLAYER(name + "_elt2", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
	n.add(NLAYER(name + "_feedfwd1", Conv, C=numG*gSize, K=ff_len, H=len, W=1));
	next_prev = n.add(NLAYER(name + "_feedfwd2", Conv, C=ff_len, K=numG*gSize, H=len, W=1));
	return n.add(NLAYER(name + "_elt3", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
}

const Network transformer = []{
	Network::lid_t enc_prev, dec_prev;
	Network::layer_set prevs;
	Network n;
	len_t numG = 8;
	len_t gSize = 64;
	len_t len = 512;
	len_t ff_len = 2048;
	len_t vocab_len = 1000;
	int nEncoder = 6;
	int nDecoder = 6;

	InputData input_enc("input_enc", fmap_shape(numG*gSize, len, 1));
	enc_prev = n.add(NLAYER("word_embed_enc", PTP, K=numG*gSize, H=len, W=1), {}, 0, {input_enc});
	for(int i=1; i<=nEncoder; ++i){
		enc_prev = add_encoder(n, "encoder"+std::to_string(i), len, numG, gSize, ff_len, enc_prev);
	}

	InputData input_dec("input_dec", fmap_shape(numG*gSize, len, 1));
	dec_prev = n.add(NLAYER("word_embed_dec", PTP, K=numG*gSize, H=len, W=1), {}, 0, {input_dec});
	for(int i=1; i<=nDecoder; ++i){
		dec_prev = add_decoder(n, "decoder"+std::to_string(i), len, numG, gSize, ff_len, dec_prev, enc_prev);
	}
	n.add(NLAYER("proj", Conv, C=numG*gSize, K=vocab_len, H=len, W=1), {dec_prev});
	return n;
}();

const Network transformer_cell = []{
	Network::lid_t enc_prev, dec_prev;
	Network::layer_set prevs;
	Network n;
	len_t numG = 8;
	len_t gSize = 64;
	len_t len = 512;
	len_t ff_len = 2048;
	len_t vocab_len = 1000;

	InputData input_enc("input_enc", fmap_shape(numG*gSize, len, 1));
	enc_prev = n.add(NLAYER("word_embed_enc", PTP, K=numG*gSize, H=len, W=1), {}, 0, {input_enc});
	enc_prev = add_encoder(n, "encoder", len, numG, gSize, ff_len, enc_prev);

	InputData input_dec("input_dec", fmap_shape(numG*gSize, len, 1));
	dec_prev = n.add(NLAYER("word_embed_dec", PTP, K=numG*gSize, H=len, W=1), {}, 0, {input_dec});
	dec_prev = add_decoder(n, "decoder", len, numG, gSize, ff_len, dec_prev, enc_prev);
	n.add(NLAYER("proj", Conv, C=numG*gSize, K=vocab_len, H=len, W=1), {dec_prev});
	return n;
}();
