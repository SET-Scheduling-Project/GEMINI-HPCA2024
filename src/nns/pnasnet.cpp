#include "nns/nns.h"

static void concat(Network::layer_set& layers, const Network::layer_set& suffix){
	layers.insert(layers.end(), suffix.begin(), suffix.end());
}

struct Genotype{
	typedef Network::layer_set (*op_func)(Network& n, const std::string& name, Network::layer_set prevs,len_t C_in, len_t C_out, len_t stride, len_t H_in, len_t H_out);
	struct entry{
		op_func op;
		int index;
	};
	entry* normal;
	int* normal_concat;
	entry* reduce;
	int* reduce_concat;
	int len, concat_len;
};

static Network::layer_set addReLUConvBN(
		Network& n, const std::string& name,
		len_t C_in, len_t C_out, len_t kernel_size,
		len_t stride, len_t H_in, len_t H_out,
		Network::layer_set prevs){
	Network::lid_t prev;
	n.add(NLAYER(name + "_relu", PTP, K=C_in, H=H_in), prevs);
	prev = n.add(NLAYER(name + "_conv", Conv, C=C_in, K=C_out, R=kernel_size, H=H_out, sH=stride));
	// BatchNorm here.
	return {prev};
}

static Network::layer_set addFactorizedReduce(
		Network& n, const std::string& name,
		len_t C_in, len_t C_out, len_t H_in, len_t H_out,
		Network::layer_set prevs){
	Network::lid_t prev, l1, l2;
	prev = n.add(NLAYER(name + "_relu", PTP, K=C_in, H=H_in), prevs);
	// These two convs have different padding, cannot merge.
	l1 = n.add(NLAYER(name + "_conv1", Conv, C=C_in, K=C_out/2, R=1, H=H_out, sH=2));
	l2 = n.add(NLAYER(name + "_conv2", Conv, C=C_in, K=C_out/2, R=1, H=H_out, sH=2), {prev});
	// BatchNorm here.
	return {l1, l2};
}

static void addCell(
		Network& n, const std::string& name, const Genotype& geno,
		len_t C_prev_prev, len_t C_prev, len_t C,
		bool reduction, int reduction_prev,
		len_t& H_prev_in, len_t& H_in,
		Network::layer_set& s0, Network::layer_set& s1){
	len_t H_out = reduction ? (H_in+1) / 2 : H_in;
	Network::layer_set curS0, curS1;
	if(reduction_prev == 1){
		curS0 = addFactorizedReduce(n, name+"_preproc0", C_prev_prev, C, H_prev_in, H_in, s0);
	}else if(reduction_prev != -1){
		curS0 = addReLUConvBN(n, name+"_preproc0", C_prev_prev, C, 1, 1, H_prev_in, H_in, s0);
	}else{
		curS0 = s0;
	}
	// TODO: here
	curS1 = addReLUConvBN(n, name+"_preproc1", C_prev, C, 1, 1, H_in, H_in, s1);

	int steps = geno.len;
	int concat_len = geno.concat_len;
	Genotype::entry* entries;
	int* concats;
	if(reduction){
		entries = geno.reduce;
		concats = geno.reduce_concat;
	}else{
		entries = geno.normal;
		concats = geno.normal_concat;
	}

	std::vector<Network::layer_set> states = {curS0, curS1};

	auto add_op = [&](const std::string& name, int idx) -> Network::layer_set {
		auto op = entries[idx].op;
		idx = entries[idx].index;
		// Add this op.
		Network::layer_set& h = states[idx];
		len_t stride = (reduction && idx < 2) ? 2 : 1;
		len_t H = (idx < 2) ? H_in : H_out;
		len_t C_in = (reduction_prev == -1 && idx == 0) ? C_prev_prev : C;
		return op(n, name, h, C_in, C, stride, H, H_out);
	};

	for(int i=0; i<steps; ++i){
		std::string stepName = name + "_" + std::to_string(i+1);
		Network::layer_set out1, out2;
		out1 = add_op(stepName + "a", 2*i);
		concat(out1, add_op(stepName + "b", 2*i+1));
		Network::lid_t out;
		out = n.add(NLAYER(name + "_elt_" + std::to_string(i), Eltwise, N=2, K=C, H=H_out), out1);
		states.push_back({out});
	}
	Network::layer_set res;
	for(int i=0; i<concat_len;++i){
		const auto& x = states[concats[i]];
		concat(res, x);
	}
	H_prev_in = H_in;
	H_in = H_out;
	s0 = s1;
	s1 = res;
}

static Network genNAS(const Genotype& geno, len_t C, len_t nClasses, unsigned nCells){
	Network::lid_t prev;
	Network n;
	len_t H_prev, H_in;
	H_in = 331;
	InputData input("input", fmap_shape(3, H_in));

	H_in = (H_in-1)/2;
	prev = n.add(NLAYER("conv0", Conv, C=3, K=96, H=H_in, R=3, sH=2), {}, 0, {input});
	// BatchNorm here.

	Network::layer_set s0, s1;
	s0 = s1 = {prev};
	H_prev = H_in;
	// Start of stem1.
	int mult = geno.concat_len;
	addCell(n, "stem1", geno, 96, 96, C/4, true, -1, H_prev, H_in, s0, s1);
	addCell(n, "stem2", geno, 96, C*mult/4, C/2, true, 1, H_prev, H_in, s0, s1);

	len_t C_prev_prev = C*mult/4;
	len_t C_prev = C*mult/2;
	len_t C_curr = C;

	bool reduction;
	int reduction_prev = 1;

	for(unsigned i=0; i<nCells; ++i){
		if(i == nCells/3 || i == 2*nCells/3){
			C_curr *= 2;
			reduction = true;
		}else{
			reduction = false;
		}
		addCell(n, "cell" + std::to_string(i+1), geno, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, H_prev, H_in, s0, s1);
		reduction_prev = reduction?1:0;
		C_prev_prev = C_prev;
		C_prev = mult*C_curr;
	}

	n.add(NLAYER("glob_relu", PTP, K = C_prev, H = H_in), s1);
	n.add(NLAYER("glob_pool", Pooling, K = C_prev, H = 1, R = H_in, sH=H_in));
	n.add(NLAYER("fc", FC, C=C_prev, K=nClasses));
	return n;
};

/*
PNASNet = Genotype(
  normal = [
	('sep_conv_5x5', 0),
	('max_pool_3x3', 0),
	('sep_conv_7x7', 1),
	('max_pool_3x3', 1),
	('sep_conv_5x5', 1),
	('sep_conv_3x3', 1),
	('sep_conv_3x3', 4),
	('max_pool_3x3', 1),
	('sep_conv_3x3', 0),
	('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
	('sep_conv_5x5', 0),
	('max_pool_3x3', 0),
	('sep_conv_7x7', 1),
	('max_pool_3x3', 1),
	('sep_conv_5x5', 1),
	('sep_conv_3x3', 1),
	('sep_conv_3x3', 4),
	('max_pool_3x3', 1),
	('sep_conv_3x3', 0),
	('skip_connect', 1),
  ],
  reduce_concat = [2, 3, 4, 5, 6],
)
*/

/*
OPS = {
  'max_pool_3x3' : lambda C_in, C_out, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1) if C_in == C_out else nn.Sequential(
	nn.MaxPool2d(3, stride=stride, padding=1),
	nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
	nn.BatchNorm2d(C_out, eps=1e-3, affine=affine)
	),
  'skip_connect' : lambda C_in, C_out, stride, affine: Identity() if stride == 1 else ReLUConvBN(C_in, C_out, 1, stride, 0, affine=affine),
  'sep_conv_3x3' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 7, stride, 3, affine=affine),
}
*/

typedef Genotype::entry entry;

static Network::layer_set addSepConv(Network& n, const std::string& name, Network::layer_set prevs,
									 len_t C_in, len_t C_out, len_t stride, len_t H_in, len_t H_out, len_t kernel_size){
	// SepConv(C_in, C_out, 3, stride, 1, affine=affine);
	Network::lid_t prev;
	n.add(NLAYER(name + "_relu", PTP, K=C_in, H=H_in), prevs);
	n.add(NLAYER(name + "_conv1", GroupConv, C=C_in, K=C_in, G=C_in, H=H_out, R=kernel_size, sH=stride));
	n.add(NLAYER(name + "_conv2", Conv, C=C_in, K=C_out, R=1, H=H_out));
	// BatchNorm here.
	// ReLU here.
	n.add(NLAYER(name + "_conv3", GroupConv, C=C_out, K=C_out, G=C_out, H=H_out, R=kernel_size));
	prev = n.add(NLAYER(name + "_conv4", Conv, C=C_out, K=C_out, R=1, H=H_out));
	// BatchNorm here.
	return {prev};
};

static Network::layer_set sep_conv_3x3(Network& n, const std::string& name, Network::layer_set prevs,
									   len_t C_in, len_t C_out, len_t stride, len_t H_in, len_t H_out){
	return addSepConv(n, name + "_sep3", prevs, C_in, C_out, stride, H_in, H_out, 3);
};

static Network::layer_set sep_conv_5x5(Network& n, const std::string& name, Network::layer_set prevs,
									   len_t C_in, len_t C_out, len_t stride, len_t H_in, len_t H_out){
	return addSepConv(n, name + "_sep5", prevs, C_in, C_out, stride, H_in, H_out, 5);
};

static Network::layer_set sep_conv_7x7(Network& n, const std::string& name, Network::layer_set prevs,
									   len_t C_in, len_t C_out, len_t stride, len_t H_in, len_t H_out){
	return addSepConv(n, name + "_sep7", prevs, C_in, C_out, stride, H_in, H_out, 7);
};

static Network::layer_set max_pool_3x3(Network& n, const std::string& name, Network::layer_set prevs,
									   len_t C_in, len_t C_out, len_t stride, len_t H_in, len_t H_out){
	(void) H_in;
	Network::lid_t prev;
	prev = n.add(NLAYER(name + "_mp3_pool", Pooling, K=C_in, H=H_out, R=3, sH=stride), prevs);
	if(C_in == C_out) return {prev};
	prev = n.add(NLAYER(name + "_mp3_conv", Conv, C=C_in, K=C_out, R=1, H=H_out));
	// BatchNorm here.
	return {prev};
}

// static constexpr Genotype::op_func avg_pool_3x3 = max_pool_3x3;

static Network::layer_set skip_connect(Network& n, const std::string& name, Network::layer_set prevs,
									   len_t C_in, len_t C_out, len_t stride, len_t H_in, len_t H_out){
	if(stride == 1){
		return prevs;
	}
	return addReLUConvBN(n, name + "_sc", C_in, C_out, 1, stride, H_in, H_out, prevs);
}

static constexpr int PNAS_geno_len = 5;
static constexpr int PNAS_conc_len = 5;
static entry PNAS_normal[2*PNAS_geno_len] = {
	{sep_conv_5x5, 0},
	{max_pool_3x3, 0},
	{sep_conv_7x7, 1},
	{max_pool_3x3, 1},
	{sep_conv_5x5, 1},
	{sep_conv_3x3, 1},
	{sep_conv_3x3, 4},
	{max_pool_3x3, 1},
	{sep_conv_3x3, 0},
	{skip_connect, 1},
};
static int normal_concat[PNAS_conc_len] = {2,3,4,5,6};
static entry PNAS_reduce[2*PNAS_geno_len] = {
	{sep_conv_5x5, 0},
	{max_pool_3x3, 0},
	{sep_conv_7x7, 1},
	{max_pool_3x3, 1},
	{sep_conv_5x5, 1},
	{sep_conv_3x3, 1},
	{sep_conv_3x3, 4},
	{max_pool_3x3, 1},
	{sep_conv_3x3, 0},
	{skip_connect, 1},
};
static int reduce_concat[PNAS_conc_len] = {2,3,4,5,6};

static Genotype PNASNet_geno = {
	.normal = PNAS_normal,
	.normal_concat = normal_concat,
	.reduce = PNAS_reduce,
	.reduce_concat = reduce_concat,
	.len = PNAS_geno_len,
	.concat_len = PNAS_conc_len,
};

const Network PNASNet = genNAS(PNASNet_geno, 216, 1001, 12);

