#include "network.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "./json/json.hpp"
#include "util.h"
#include "coremapping.h"

using json = nlohmann::json;
using string = std::string;

const Network* network=nullptr;

InputData::InputData(const std::string& _name, const fmap_shape& _data_shape)
	:name(_name), data_shape(_data_shape){
	assert(data_shape.size > 0);
}

const fmap_shape& InputData::get_shape() const{
	return data_shape;
}

Node::Node(const Layer* _l, const Bitset& _ifmPrevs, len_t _external_C, bwidth_t width, const Bitset& _wgtPrevs, lid_t _id)
	:l(_l), ifmPrevs(_ifmPrevs), wgtPrevs(_wgtPrevs), prevs(_ifmPrevs | _wgtPrevs), external_C(_external_C), id(_id){
	if(width > 0) const_cast<Layer*>(_l)->set_bitwidth(width);
}

const Layer& Node::layer() const{
	return *l;
}

const std::string& Node::name() const{
	return l->get_name();
}

Node::lid_t Node::getid() const{
	return id;
}

const Bitset& Node::getIfmPrevs() const{
	return ifmPrevs;
}

const Bitset& Node::getWgtPrevs() const{
	return wgtPrevs;
}

const Bitset& Node::getPrevs() const{
	return prevs;
}

bool Node::hasWgtPrevs() const{
	return wgtPrevs.count() > 0;
}

const Bitset& Node::get_nexts() const{
	return nexts;
}

utime_t Node::get_utime() const{
	return l->get_utime();
}

len_t Node::get_external_C() const{
	return external_C;
}
/*
void Node::ifm_to_prev_ofm(fmap_range& ifm_rng) const{

}
*/
void Node::add_next(Node::lid_t l){
	nexts.set(l);
}


Network::Network(){
	// acc_ops.push_back(0);
	/*
	acc_ops[0] = 0;
	for(size_t i=0;i<len;++i){
		acc_ops[i+1] = acc_ops[i] + layers[i].total_op();
	}
	*/
}

void Network::err_mismatch(const std::string& lname, const fmap_shape& shape1, const fmap_shape& shape2, bool total){
	std::cerr << "The h*w of inputs of Layer " << lname << " mismatch!" << std::endl;
	if(total){
		std::cerr << "\t Real ifmap: " << shape1 << ", Total ifmap: " << shape2 << '.'<< std::endl;
	}else{
		std::cerr << "\tH*W: (" << shape1.h << ',' << shape1.w << ") with (" << shape2.h << ',' << shape2.w << ")."<< std::endl;
	}
	throw std::logic_error("Input dimension mismatch.");
}

void Network::err_eltwise(const std::string& lname, const len_t from_C, const len_t add_C, const len_t elt_C){
	std::cerr << "The channel of inputs of EltwiseLayer " << lname << " mismatch!" << std::endl;
	std::cerr << "\t C: (" << from_C << ", " << from_C + add_C << ") should not include multiples of " << elt_C << '.'<< std::endl;
	throw std::logic_error("Eltwise ifmap channel mismatch.");
}

Network::lid_t Network::add(const Layer* l, const layer_set& ifmPrevs, bwidth_t width, std::vector<InputData> ext_data, const layer_set& wgtPrevs){
	// If no prevs indicated, use default_bs.
	bool default_prev = (ext_data.empty() && ifmPrevs.empty());
	lid_t last_id=0;
	Bitset prev_layers, prevWgts;
	if(default_prev){
		assert(!layers.empty());
		assert(wgtPrevs.empty());
		last_id = static_cast<lid_t>(layers.size()-1);
		prev_layers.set(last_id);
	}else{
		for(const lid_t& i: ifmPrevs) prev_layers.set(i);
		for(const lid_t& i: wgtPrevs) prevWgts.set(i);
	}

	fmap_shape padded_ifm;
	padded_ifm.c = 0;

	// TODO: remove constraint for eltwise layer.
	len_t eltwise_C = 0;
	if(IS_INSTANCE(l, EltwiseLayer)){
		eltwise_C = static_cast<const EltwiseLayer*>(l)->ofmap_shape().c;
	}

	auto check_func = [&](const fmap_shape& in_shape){
		if(eltwise_C > 0 && (padded_ifm.c % eltwise_C) + in_shape.c > eltwise_C){
			err_eltwise(l->get_name(), padded_ifm.c, in_shape.c, eltwise_C);
		}
		if(padded_ifm.c == 0){
			padded_ifm = in_shape;
		}else if(padded_ifm.h == in_shape.h && padded_ifm.w == in_shape.w){
			padded_ifm.c += in_shape.c;
		}else{
			std::cout<<"Error throwed by check_func" <<std::endl;
			err_mismatch(l->get_name(), padded_ifm, in_shape);
		}
	};

	for(const InputData& input : ext_data){
		const fmap_shape& in_shape = input.get_shape();
		check_func(in_shape);
	}

	// The number of external ifmap channels
	len_t external_C = padded_ifm.c;
	if(eltwise_C > 0 && external_C > eltwise_C){
		err_eltwise(l->get_name(), 0, external_C, eltwise_C);
	}
	FOR_BITSET(it, prev_layers){
		lid_t layer_id = it;
		const fmap_shape& in_shape=getNode(layer_id).layer().ofmap_shape();
		check_func(in_shape);
	}

	padded_ifm.update_size();
	if(!const_cast<Layer*>(l)->set_padded_ifm(padded_ifm)){
		err_mismatch(l->get_name(), padded_ifm, l->tot_ifmap_shape(), true);
	}

	if(prevWgts.count() > 0){
		assert(l->weight_size() > 0);
		fmap_shape wgtShape = l->weight_shape();
		len_t curC = 0;

		FOR_BITSET(it, prevWgts){
			lid_t layer_id = it;
			const fmap_shape& out_shape=getNode(layer_id).layer().ofmap_shape();
			if(out_shape.h != wgtShape.h || out_shape.w != wgtShape.w){
				err_mismatch(l->get_name(), out_shape, wgtShape);
			}
			curC += out_shape.c;
		}

		if(curC != wgtShape.c){
			fmap_shape curShape = wgtShape;
			curShape.c = curC;
			err_mismatch(l->get_name(), curShape, wgtShape, true);
		}
	}

	if(layers.size() >= std::numeric_limits<lid_t>::max()){
		throw std::overflow_error("Too many layers! Consider using a larger format for lid_t (perhaps uint32_t?)");
	}

	if(layers.size() >= prev_layers.max_size()){
		throw std::overflow_error("Too many layers! Consider using a larger Bitset (perhaps 4096?)");
	}

	lid_t cur_id = static_cast<lid_t>(layers.size());
	FOR_BITSET(it, prev_layers){
		lid_t layer_id = it;
		layers[layer_id].add_next(cur_id);
	}
	FOR_BITSET(it, prevWgts){
		lid_t layer_id = it;
		layers[layer_id].add_next(cur_id);
	}
	layers.emplace_back(l, prev_layers, external_C, width, prevWgts, cur_id);

	return cur_id;
}

const Node& Network::getNode(lid_t id) const{
	return layers[id];
}

const Node& Network::operator[](Network::lid_t id) const{
	return layers[id];
}

bool Network::has_dep(const Bitset& src, const Bitset& dst) const{
	lid_t layer_id;
	if(src.count()>3){
		// O(C*d)
		Bitset b;
		FOR_BITSET(it, dst){
			layer_id = it;
			b |= getNode(layer_id).getPrevs();
		}
		FOR_BITSET(it, src){
			if(b.contains(it)) return true;
		}
	}else{
		// O(d*s)
		lid_t srcs[3];
		size_t src_num = 0;
		FOR_BITSET(it, src){
			srcs[src_num++]=it;
		}
		FOR_BITSET(it, dst){
			const Bitset& dst_set=getNode(it).getPrevs();
			for(size_t i=0; i<src_num; ++i){
				if(dst_set.contains(srcs[i])) return true;
			}
		}
	}
	return false;
}

void Network::set_utime(const CoreMapper& mapper) const{
	for(const Node& n:layers){
		Layer& l = const_cast<Layer&>(n.layer());
		mapper.set_utime(l);
	}
}

Network::lid_t Network::len() const{
	return static_cast<lid_t>(layers.size());
}

bool Network::is_chain() const{
	for(size_t i=1;i<layers.size();++i){
		if(!layers[i].getPrevs().contains(i-1)) return false;
	}
	return true;
}

void Network::cal_not_conv(Bitset& not_conv_layers) const{
	for (int t = 0; t < layers.size();t++) {
		if (!REF_IS_INSTANCE(getNode(t).layer(), ConvLayer)) {
			not_conv_layers.set(t);
		}
	}
}
