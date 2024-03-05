#include "ltreenode.h"

#include <cassert>

LTreeNode::LTreeNode(const Bitset& _layer_set, len_t _num_batch, LTreeNode *_parent, NodeType _t)
	:t((_t == NodeType::L&&_layer_set.count()>1)?(_parent->t == NodeType::S? NodeType::T : NodeType::S):_t),
	 isNewNode(true), parent(_parent), layer_set(_layer_set), num_batch(_num_batch){
	if(_parent) _parent->add(this);
}

LTreeNode::LTreeNode(lid_t _layer, len_t _num_batch, LTreeNode *_parent)
	:t(NodeType::L), isNewNode(true), parent(_parent), layer_set(Bitset(_layer)), num_batch(_num_batch){
	if(_parent) _parent->add(this);
}

void LTreeNode::add(LTreeNode* child){
	children.push_back(child);
	// Add to list if needed.
}

const LTreeNode::node_vec& LTreeNode::get_children(){
	return children;
}

void LTreeNode::set_parent(LTreeNode* _parent) {
	parent = _parent;
}

LTreeNode::NodeType LTreeNode::get_type(){
	return t;
}

const Bitset& LTreeNode::layers(){
	return layer_set;
}

void LTreeNode::init_root(){
	traverse_lset();
	traverse();
}

void LTreeNode::confirm(){
	isNewNode = false;
	modified = false;
	for(auto it: children){
		it->confirm();
	}
}

bool LTreeNode::isModified() const{
	return modified;
}

bool LTreeNode::isNew() const{
	return isNewNode;
}

utime_t LTreeNode::get_utime() const{
	return unit_time;
}

len_t LTreeNode::get_bgrp_num() const{
	return num_bgrp;
}

len_t LTreeNode::get_bgrp_size() const{
	return num_batch / num_bgrp;
}

len_t LTreeNode::get_tot_batch() const{
	return num_batch;
}

const std::vector<lid_t>& LTreeNode::get_stages() const{
	return stage;
}

lid_t LTreeNode::get_num_stage() const{
	return num_stage;
}

bool LTreeNode::get_to_dram() const{
	return to_dram;
}

const Bitset& LTreeNode::get_dirp_set() const{
	return dirp_set;
}

void LTreeNode::reset_lset(){
	layer_set.clear();
}

LTreeNode::~LTreeNode(){
	for(auto child: children){
		if(child == this){
			assert(false);
		}
		delete child;
	}
}

LTreeNode* LTreeNode::copy() const{
	LTreeNode * l = new LTreeNode(*this);
	l->children.clear();
	LTreeNode* c;
	for(auto child:children){
		c = child->copy();
		c->parent = l;
		l->children.push_back(c);
	}
	return l;
}

bool LTreeNode::is_shortcut(lid_t from_id, const LTreeNode& to){
	/*
	std::cout << "SCT ";
	std::cout << network->getNode(from_id).name() << ' ';
	std::cout << network->getNode(to.layer_set.first()).name() << std::endl;
	*/
	const LTreeNode* lca = to.parent;
	const LTreeNode* to_child = &to;
	while(!lca->layer_set.contains(from_id)){
		to_child = lca;
		lca = lca->parent;
		assert(lca);
	}
	bool found = false;
	lid_t from_pos=0, to_pos=0;
	lid_t i=static_cast<lid_t>(lca->children.size());
	while (i-->0) {
		if(lca->children[i] == to_child){
			found=true;
			to_pos=i;
			break;
		}
	}
	assert(found);
	found = false;
	while (i-->0) {
		const Bitset& cur_set = lca->children[i]->layer_set;
		if(cur_set.contains(from_id)){
			found=true;
			from_pos=i;
			break;
		}
	}
	assert(found);
	bool is_sc=(lca->parent == nullptr && lca->t == NodeType::T);
	if(lca->t == NodeType::S){
		is_sc |= (lca->stage[to_pos] > lca->stage[from_pos]+1);
	}else{
		assert(lca->t == NodeType::T);
		is_sc |= to_pos > from_pos+1;
	}
	if(is_sc){
		LTreeNode* from_node=lca->children[from_pos];
		while (from_node->t != NodeType::L) {
			for(auto* n: from_node->children){
				if(n->layer_set.contains(from_id)){
					from_node = n;
					break;
				}
			}
		}
		from_node->to_dram = true;
	}
	//std::cout << "case2 " << is_sc << ' ' << (int)from_pos << ' ' << (int)to_pos << std::endl;
	return is_sc;
}

void LTreeNode::traverse(bool batch_update_only){
	//std::cout << children.size() << '#' << std::endl;

	if(t == NodeType::L){
		assert(layer_set.count() == 1);
		const Node& n = network->getNode(layer_set.first());
		unit_time = n.get_utime();
		num_bgrp = 1;
		if (!batch_update_only) {
			to_dram = (n.get_nexts().count() == 0);
		}
		dirp_set.clear();
		height = 0;
		const Bitset& prevs = n.getPrevs();
		FOR_BITSET(it, prevs){
			lid_t prev = it;
			if(!is_shortcut(prev, *this)){
				dirp_set.set(prev);
			}
		}
		return;
	}

	assert(children.size() > 0);
	len_t child_batch = children.front()->num_batch;
	assert(num_batch % child_batch == 0);
	num_bgrp = num_batch / child_batch;

	unit_time = 0;
	height = 0;
	for(auto child: children){
		assert(child->num_batch == child_batch);
		child->traverse(batch_update_only);
		unit_time += child->unit_time;
		height = MAX(height, child->height + 1);
	}
	if(t == NodeType::S) unit_time = (unit_time * (num_bgrp + num_stage)) / num_bgrp;
}

void LTreeNode::traverse_lset(bool calc_type){
	calc_type |= (t == NodeType::L) && (children.size() > 0);
	if(calc_type){
		if(children.size() == 0){
			t = NodeType::L;
		}else{
			assert(parent != nullptr);
			t = (parent->t == NodeType::S) ? NodeType::T : NodeType::S;
		}
	}

	bool calc_lset = (layer_set.count() == 0);

	// Check if we need to calculate each subset's stage.
	bool calc_stage = false;
	if(t == NodeType::S){
		if(calc_lset || stage.size() != children.size()){
			stage.clear();
			stage.reserve(children.size());
			num_stage = 0;
			calc_stage = true;
		}
	}else{
		stage.clear();
		num_stage = 0;
	}

	modified = isNewNode;
	for(auto child: children){
		child->traverse_lset(calc_type);
		if(calc_lset) layer_set |= child->layer_set;
		// Calc child's stage.
		if(calc_stage){
			lid_t cur_stage = 0;
			size_t cur_pos = 0;
			for(auto prev_ch: children){
				if(prev_ch == child) break;
				lid_t last_stage = stage[cur_pos++];
				if(last_stage < cur_stage) continue;
				if(network->has_dep(prev_ch->layer_set, child->layer_set)){
					cur_stage = last_stage + 1;
					num_stage = MAX(num_stage, cur_stage);
				}
			}
			stage.push_back(cur_stage);
		}
		modified |= child->modified;
	}
}

