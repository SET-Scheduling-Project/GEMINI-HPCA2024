#ifndef NETWORK_H
#define NETWORK_H

#include <cstdint>
#include <memory>
#include <vector>
#include "bitset.h"
#include "layer.h"
#include "util.h"

class CoreMapper;
//#include "coremapping.h"

class InputData{
private:
	std::string name;
	fmap_shape data_shape;
public:
	InputData(const std::string& _name, const fmap_shape& _data_shape);
	const fmap_shape& get_shape() const;
	~InputData()=default;
};

/*
 * Here we use Node instead of directly using Layer, since
 * 1. There will be many kinds of derived class from layer (Conv, Pool, ...)
 *        where a single Node class is more friendly to Network.
 * 2. In the future one node may contain several layers,
 *        which means in this way we can maintain compatibility.
 */

class Node{
	// Eltwise(n): channel n->1
	/* Pooling(core h*w, stride s*t, offset oh*ow):
	 * output [x1,x2)x[y1,y2)
	 * ->[oh+x1*s,oh+x2*s+h)x[ow+y1*t,ow+y2*t+w)
	 */
	/*
	struct PostProc{
		union{
			len_t h,w,s,t,oh,ow;
			len_t n;
		}u;
	};
	*/
public:
	typedef std::uint16_t lid_t;
	// Not used, may be used later.
	typedef std::vector<lid_t> layer_set;
private:
	std::unique_ptr<const Layer> l;
	//const Layer* l;
	const Bitset ifmPrevs;
	const Bitset wgtPrevs;
	const Bitset prevs;
	const lid_t id;
	//const layer_set prev_order;
	Bitset nexts;
	len_t external_C;
public:
	Node(const Layer* _l, const Bitset& _ifmPrevs, len_t _external_C, bwidth_t width = 0, const Bitset& _wgtPrevs = {}, lid_t _id = -1);
	Node(const Node& n) = delete;
	Node(Node&& n)=default;
	const Layer& layer() const;
	const std::string& name() const;
	lid_t getid() const;
	const Bitset& getIfmPrevs() const;
	const Bitset& getWgtPrevs() const;
	const Bitset& getPrevs() const;
	bool hasWgtPrevs() const;
	//const std::vector<lid_t>& get_prev_order() const;
	const Bitset& get_nexts() const;
	utime_t get_utime() const;
	len_t get_external_C() const;
	//void ifm_to_prev_ofm(fmap_range& ifm_rng) const;
	void add_next(lid_t l);
	~Node()=default;
};

class Network{
public:
	typedef Node::lid_t lid_t;
	typedef Node::layer_set layer_set;
	/*struct link{
		lid_t from;
	};*/
private:
	std::vector<InputData> inputs;
	std::vector<Node> layers;
	[[noreturn]] void err_mismatch(const std::string& lname, const fmap_shape& shape1, const fmap_shape& shape2, bool total=false);
	[[noreturn]] void err_eltwise(const std::string& lname, const len_t from_C, const len_t add_C, const len_t elt_C);
public:
	Network();
	Network(const Network& n)=delete;
	Network(Network&& n)=default;
	// Add ", const std::vector<lid_t>& prev_order" if needed.
	lid_t add(const Layer* l, const layer_set& ifmPrevs={}, bwidth_t width=0, std::vector<InputData> ext_data={}, const layer_set& wgtPrevs={});
	const Node& getNode(lid_t id) const;
	//access_t total_op(lid_t from, lid_t to) const;
	const Node& operator[](lid_t id) const;
	bool has_dep(const Bitset& src, const Bitset& dst) const;
	void set_utime(const CoreMapper& mapper) const;
	lid_t len() const;
	bool is_chain() const;
	void cal_not_conv(Bitset& not_conv_layers) const;
	~Network()=default;
};

extern const Network* network;

#endif // NETWORK_H
