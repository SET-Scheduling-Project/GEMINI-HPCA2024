#include "nns/nns.h"
#include <cstdarg>
#include <cstdio>
#include <string>
#include <exception>

static std::string fmt(const char* format, ...) {
	va_list args;
	va_start(args, format);
	int s_len = std::vsnprintf(nullptr, 0, format, args) + 1; // Extra space for '\0'
	va_end(args);
	if(s_len <= 0){
		throw std::runtime_error(std::string("Error during formatting ")+format);
	}
	char* buf = new char[s_len];
	va_start(args, format);
	std::vsnprintf(buf, s_len, format, args);
	va_end(args);
	std::string s(buf);
	delete[] buf;
	return s;
}

static Network::lid_t Stem(Network& n) {
	InputData input("input", fmap_shape(3, 299));
	n.add(NLAYER("Conv_1a_3x3", Conv, C=3, K=32, R=3, H=149, sH=2), {}, 0, {input});
	n.add(NLAYER("Conv_2a_3x3", Conv, C=32, R=3, H=147));
	n.add(NLAYER("Conv_2b_3x3", Conv, C=32, K=64, R=3, H=147));
	n.add(NLAYER("MaxPool_3a_3x3", Pooling, K=64, H=73, R=3, sH=2));
	n.add(NLAYER("Conv_3b_1x1", Conv, C=64, K=80, H=73));
	n.add(NLAYER("Conv_4a_3x3", Conv, C=80, K=192, H=71, R=3));
	return n.add(NLAYER("Conv_4b_3x3", Conv, C=192, K=256, H=35, R=3, sH=2));
}

static Network::lid_t Inception_Resnet_A(Network& n, Network::lid_t lastid, int block_no) {
	auto branch1 = n.add(NLAYER(fmt("A%d_Conv_1a_1x1",block_no), Conv, C=256, K=32, H=35));
	n.add(NLAYER(fmt("A%d_Conv_2a_1x1",block_no), Conv, C=256, K=32, H=35), {lastid});
	auto branch2 = n.add(NLAYER(fmt("A%d_Conv_2b_3x3",block_no), Conv, C=32, K=32, H=35, R=3));
	n.add(NLAYER(fmt("A%d_Conv_3a_1x1",block_no), Conv, C=256, K=32, H=35), {lastid});
	n.add(NLAYER(fmt("A%d_Conv_3b_3x3",block_no), Conv, C=32, K=32, H=35, R=3));
	auto branch3 = n.add(NLAYER(fmt("A%d_Conv_3c_3x3",block_no), Conv, C=32, K=32, H=35, R=3));
	auto conv4 = n.add(NLAYER(fmt("A%d_Conv_4_1x1",block_no), Conv, C=96, K=256, H=35), {branch1,branch2,branch3});
	return n.add(NLAYER(fmt("A%d_Eltwise",block_no), Eltwise, N=2, K=256, H=35), {lastid, conv4});
}

static Network::layer_set Reduction_A(Network& n, Network::lid_t lastid){
	auto branch1 = n.add(NLAYER("RA_Conv_1a_3x3", Conv, C=256, K=384, H=17, R=3, sH=2));
	n.add(NLAYER("RA_Conv_2a_1x1", Conv, C=256, K=192, H=35), {lastid});
	n.add(NLAYER("RA_Conv_2b_3x3", Conv, C=192, K=192, H=35, R=3));
	auto branch2 = n.add(NLAYER("RA_Conv_2c_3x3", Conv, C=192, K=256, H=17, R=3, sH=2));
	auto branch3 = n.add(NLAYER("RA_MaxPool_3a_3x3", Pooling, K=256, H=17, R=3, sH=2), {lastid});
	return {branch1, branch2, branch3};
}

static Network::lid_t Inception_Resnet_B(Network& n, Network::layer_set prev_layer, int block_no){
	auto branch1 = n.add(NLAYER(fmt("B%d_Conv_1a_1x1", block_no), Conv, C=896, K=128, H=17), prev_layer);
	n.add(NLAYER(fmt("B%d_Conv_2a_1x1",block_no), Conv, C=896, K=128, H=17), prev_layer);
	n.add(NLAYER(fmt("B%d_Conv_2b_1x7",block_no), Conv, C=128, K=128, H=17, R=1, S=7));
	auto branch2 = n.add(NLAYER(fmt("B%d_Conv_2c_7x1",block_no), Conv, C=128, K=128, H=17, R=7, S=1));
	auto conv3 = n.add(NLAYER(fmt("B%d_Conv_3_1x1", block_no), Conv, C=256, K=896, H=17), {branch1, branch2});
	prev_layer.push_back(conv3);
	return n.add(NLAYER(fmt("B%d_Eltwise", block_no), Eltwise, N=2, K=896, H=17), prev_layer);
}

static Network::layer_set Reduction_B(Network& n, Network::lid_t lastid){
	auto branch1 = n.add(NLAYER("RB_MaxPool_1a_3x3", Pooling, K=896, H=8, R=3, sH=2));
	n.add(NLAYER("RB_Conv_2a_1x1", Conv, C=896, K=256, H=17), {lastid});
	auto branch2 = n.add(NLAYER("RB_Conv_2b_3x3", Conv, C=256, K=384, H=8, R=3, sH=2));
	n.add(NLAYER("RB_Conv_3a_1x1", Conv, C=896, K=256, H=17), {lastid});
	auto branch3 = n.add(NLAYER("RB_Conv_3b_3x3", Conv, C=256, K=256, H=8, R=3, sH=2));
	n.add(NLAYER("RB_Conv_4a_1x1", Conv, C=896, K=256, H=17), {lastid});
	n.add(NLAYER("RB_Conv_4b_3x3", Conv, C=256, K=256, H=17, R=3));
	auto branch4 = n.add(NLAYER("RB_Conv_4c_3x3", Conv, C=256, K=256, H=8, R=3, sH=2));
	return {branch1,branch2,branch3,branch4};
}

static Network::lid_t Inception_Resnet_C(Network& n, Network::layer_set prev_layer, int block_no){
	auto branch1 = n.add(NLAYER(fmt("C%d_Conv_1a_1x1",block_no), Conv, C=1792, K=192, H=8), prev_layer);
	n.add(NLAYER(fmt("C%d_Conv_2a_1x1",block_no), Conv, C=1792, K=192, H=8), prev_layer);
	n.add(NLAYER(fmt("C%d_Conv_2b_1x3",block_no), Conv, C=192, K=192, H=8, R=1, S=3));
	auto branch2 = n.add(NLAYER(fmt("C%d_Conv_2c_3x1",block_no), Conv, C=192, K=192, H=8, R=3, S=1));
	auto conv3 = n.add(NLAYER(fmt("C%d_Conv_3_1x1",block_no), Conv, C=384, K=1792, H=8), {branch1, branch2});
	prev_layer.push_back(conv3);
	return n.add(NLAYER(fmt("C%d_Eltwise",block_no), Eltwise, N=2, K=1792, H=8), prev_layer);
}

const Network inception_resnet_v1 = []{
	Network n;
	Network::lid_t last_id = Stem(n);
	for (int i=1; i<=5; ++i) {
		last_id = Inception_Resnet_A(n, last_id, i);
	}
	Network::layer_set last_ids = Reduction_A(n, last_id);
	for (int i=1; i<=10; ++i) {
		if (i==1) {
			last_id = Inception_Resnet_B(n, last_ids, i);
		} else {
			last_id = Inception_Resnet_B(n, {last_id}, i);
		}
	}
	last_ids = Reduction_B(n, last_id);
	for (int i=1; i<=5; ++i) {
		if (i==1) {
			last_id = Inception_Resnet_C(n, last_ids, i);
		} else {
			last_id = Inception_Resnet_C(n, {last_id}, i);
		}
	}
	n.add(NLAYER("AvgPool", Pooling, K=1792, R=8, H=1));
	n.add(NLAYER("Dropout", FC, C=1792, K=1792));
	n.add(NLAYER("FC", FC, C=1792, K=1000));
	return n;
}();

