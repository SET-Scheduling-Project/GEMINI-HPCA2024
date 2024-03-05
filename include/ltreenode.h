#ifndef LTREENODE_H
#define LTREENODE_H

#include "util.h"
#include "bitset.h"
#include "network.h"
#include <vector>

typedef uint16_t lid_t;
typedef std::uint32_t len_t;
typedef std::int16_t cidx_t;
typedef double utime_t;
typedef std::uint64_t vol_t;

class SAEngine;

class LTreeNode{
public:
	enum class NodeType : std::uint8_t{
		S,T,L
	};
	typedef std::vector<LTreeNode*> node_vec;
	bool isNewNode, modified;
private:
	NodeType t;
	
	LTreeNode* parent;
	node_vec children;
	Bitset layer_set;
	utime_t unit_time;
	len_t num_bgrp, num_batch;
	// Only for S Cut.
	std::vector<lid_t> stage;
	// TODO: actually this is max_stage.
	lid_t num_stage;
	// Only for Layer.
	bool to_dram;
	// Direct prevs.
	Bitset dirp_set;

	len_t height;


	// Also sets to_dram.
	static bool is_shortcut(lid_t from_id, const LTreeNode& to);
	void traverse(bool batch_update_only=false);
	void traverse_lset(bool calc_type = false);
public:
	LTreeNode(const Bitset& _layer_set, len_t _num_batch, LTreeNode* _parent=nullptr, NodeType _t=NodeType::L);
	LTreeNode(lid_t _layer, len_t _num_batch, LTreeNode* _parent=nullptr);
	LTreeNode(const LTreeNode& node)=default;
	void add(LTreeNode* child);
	const node_vec& get_children();
	LTreeNode* remove_last_child();
	void insert_child_at_front(LTreeNode *node);
	void set_parent(LTreeNode* _parent);
	NodeType get_type();
	const Bitset& layers();
	void init_root();
	void confirm();
	bool isModified() const;
	bool isNew() const;
	utime_t get_utime() const;
	len_t get_bgrp_num() const;
	len_t get_bgrp_size() const;
	len_t get_tot_batch() const;
	//lid_t get_nstages() const;
	const std::vector<lid_t>& get_stages() const;
	lid_t get_num_stage() const;
	bool get_to_dram() const;
	const Bitset& get_dirp_set() const;
	void reset_lset();
	~LTreeNode();

	friend class SAEngine;
	friend void halv_bat(LTreeNode* node);
	friend void flat_bat(LTreeNode* node, len_t n_batch);
	friend class Segment_scheme;
	LTreeNode* copy() const;
};

#endif // LTREENODE_H
