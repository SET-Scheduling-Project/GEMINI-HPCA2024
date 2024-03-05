#ifndef SCHNODE_H
#define SCHNODE_H

#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <vector>


#include "bufferusage.h"
#include "cluster.h"
#include "coremapping.h"
#include "ltreenode.h"
#include "network.h"
#include "noc.h"
#include "placement.h"
#include "util.h"
#include "spatial_mapping/light_placement.h"

class LayerEngine;
//#include "layerengine.h"

// TODO: Temp. Remove later.
typedef Network::lid_t lid_t;

class Cut;
namespace Json{
	class Value;
};


class SchNode{
public:
	typedef SchNode* sn_ptr;
	typedef const SchNode* csn_ptr;
	typedef SchNode& sn_ref;
	typedef std::deque<sn_ptr> sn_vec;
	typedef std::unordered_map<lid_t, LNode*> nodeList_t;
	typedef LTreeNode::NodeType NodeType;

	typedef Cut* cut_ptr;
	typedef std::uint32_t wlid_t;
	typedef std::uint32_t tfid_t;
	typedef std::uint32_t jsonindex_t;
	static bw_t DRAM_bw;
	static vol_t ubuf;
	static bool cluster_base;
	struct SchCost{
		energy_t energy;
		cycle_t time;
		SchCost(energy_t _energy=energy_inf, cycle_t _time=0);
		cost_t cost(len_t nbatch=1) const;
		SchCost& operator+=(const SchCost& other);
		SchCost& operator*=(len_t other);
		bool operator!=(const SchCost& other) const;
		bool isValid() const;
		friend std::ostream& operator<<(std::ostream& os, const SchCost& cost);
	};
	struct energy_breakdown {
		energy_t ubuf;
		energy_t buf;
		energy_t bus;
		energy_t mac;
		energy_t NoC_hop_cost;
		energy_t NoP_hop_cost;
		energy_t DRAM_cost;
	};
protected:
	bool valid;
	const NodeType type;
	//const lid_t from, to;
	const len_t num_batch;
	bool is_base;
	bool calc_noc;
	len_t wgt_bgrp; // Size of weight batch_group.
	const Cluster cluster;
	const Cut* const parent;
	SchCost cost;
	NoC noc;
	BufferUsage buf_usage, ifm_usage, wgt_usage;
	vol_t  ifm_buf, wgt_buf,fmap_buf;//used for cal buffer
	vol_t tot_data;//used for core_mul_data
	energy_t ubuf_energy, buf_energy, bus_energy, mac_energy;
	access_t dram_access;
	vol_t core_mul_data;
	nodeList_t* const lnodeList;

	//const sn_vec* cur_vec;
	static wlid_t workload_cnt;
	static tfid_t transferid_cnt;
	static std::vector<std::vector<std::vector<jsonindex_t> > > wlid;
	static std::vector<bool> from_core, weight_from_core, to_dram;
	static std::vector<std::map<fmap_range, jsonindex_t> > ofmapid; 
	static std::vector<std::set<Json::Value> > curr_ifmap;
	static std::vector<std::set<Json::Value> > curr_weight;
	static std::map<std::string,lid_t> name_to_id;
	static Json::Value DRAM;
	static std::map<Json::Value,jsonindex_t> DRAM_ofmap_pos;
	static std::map<Json::Value,jsonindex_t> DRAM_weight_pos;
	static std::map<tfid_t,jsonindex_t> DRAM_ifmap_pos;
	static csn_ptr root;

	SchNode(const SchNode& node) = default;
public:
	// For root only.
	// SchNode(NodeType t, len_t _bgrp, const Cluster& _c);
	SchNode(NodeType t, const Cluster& _c, cut_ptr _parent, len_t nbatch);
	//SchNode(lt_ptr lt);
	//virtual SchCost search(lid_t depth)=0;
	virtual ~SchNode() =0;
	virtual bool contains(lid_t layerid) const =0;
	virtual SchNode* copy(Cut* newParent = nullptr) const =0;
	virtual void searchInc(LTreeNode* node, bool fast_mode, bool usefillin = false, const Light_placement& place = Light_placement()) = 0;
	void setParent(Cut* newParent);
	bool is_valid() const;
	SchCost get_cost() const;
	const NoC& get_noc() const;
	const Cluster& getCluster() const;
	NodeType get_type() const;
	const BufferUsage& get_buf_usage() const;
	const BufferUsage& get_ifm_usage() const;
	const BufferUsage& get_wgt_usage() const;
	const vol_t get_ifm() const;
	const vol_t get_wgt() const;
	const vol_t get_fmap() const;
	const vol_t get_core_mul_data() const;
	const vol_t get_data() const;
	const len_t get_num_batch() const;
	energy_t get_ubuf_energy() const;
	energy_t get_buf_energy() const;
	energy_t get_bus_energy() const;
	energy_t get_mac_energy() const;
	access_t get_dram_access()const;
	void write_energy_record() const;
	
	//sn_ptr newNode(SchNode::NodeType t, const Bitset& layers, len_t _bgrp, const Cluster& _c);
	bool is_DRAM_cut() const;

	//static SchNode* search_all(lid_t from, lid_t to, len_t bgrp);
	//static sn_ptr root;
	static LayerEngine* layerMapper;
	static LayerEngine* layerMapper_fast;
	static len_t tot_batch;
	static energy_breakdown record;

	virtual void print_struct(std::string pad = "", std::ostream& os = std::cout) const =0;
	void print_res(std::ostream& os = std::cout) const;
	friend std::ostream& operator<<(std::ostream& os, const SchNode& sch);
	friend std::ostream& operator<<(std::ostream& os, const SchNode* sch);

	Json::Value IR_gen() const;
	virtual void add_workload_and_dfs(len_t batch_offset, len_t segment, std::vector<Json::Value>& workload_list) const = 0;
	virtual const LNode* get_lnode_by_id(lid_t id) const = 0;
};

class LNode : public SchNode{
	friend class StdLayerEngine;
	friend class FastLayerEngine;
private:
	lid_t layerid;
	const Node& layert;
	PlaceSch place_sch;
	const Bitset dirp_set;
	const bool to_dram;
	mlen_t nearest_dram_id;
	CoreMapper::CoreMapping tileSch;
	bool search(bool usefillin = false, const Light_placement &place = Light_placement());
	bool search_fast();
public:
	//LNode(lid_t _layerid, const Cluster& _c, cut_ptr _parent, len_t nbatch);
	LNode(LTreeNode* _node, const Cluster& _c, cut_ptr _parent, const Light_placement& place, bool fast_mode=false,bool base=false, bool calc_noc = true);
	LNode(LTreeNode* _node, const Cluster& _c, cut_ptr _parent,bool fast_mode=false,bool base = false, bool calc_noc = true);
	//LNode(const Bitset& _layers, len_t _bgrp, const Cluster& _c, csn_ptr _parent);
	virtual SchNode* copy(Cut* newParent = nullptr) const override;
	virtual void searchInc(LTreeNode* node, bool fast_mode, bool usefillin = false, const Light_placement& place = Light_placement()) override;
	void searchLayer(bool usefillin = false, const Light_placement &place = Light_placement());
	void searchLayer_fast();
	//virtual SchCost search(lid_t depth) override;
	virtual bool contains(lid_t _layerid) const override;
	const Node& getLayer() const;
	const PlaceSch& get_place_sch() const;
	const Bitset& get_dirp_set() const;
	bool get_to_dram() const;
	bool is_seg() const;
	mlen_t get_nearest_dram() const;
	virtual ~LNode() override;

	static const Cut* get_lca(const LNode* node1, const LNode* node2);

	virtual void print_struct(std::string pad = "", std::ostream& os = std::cout) const override;

	virtual void add_workload_and_dfs(len_t batch_offset, len_t segment, std::vector<Json::Value>& workload_list) const override;
	virtual const LNode* get_lnode_by_id(lid_t id) const override;
};

class Cut : public SchNode{
	sn_vec oldChildren;
	LTreeNode* curNode;
protected:
	const Bitset layers;
	sn_vec children;
	len_t num_bgrp;
	void construct(LTreeNode* node, const std::vector<Light_placement>& place);
	virtual void construct(LTreeNode* node, bool usefillin, const Light_placement& place) = 0;
	virtual void construct_fast(LTreeNode* node) = 0;
public:
	//Cut(NodeType t, const Bitset& _layers, const Cluster& _c, csn_ptr _parent, len_t nbgrp, len_t nbatch);
	Cut(NodeType t, LTreeNode* node, const Cluster& _c, cut_ptr _parent);
	//virtual SchCost search(lid_t depth) override =0;
	virtual SchNode* copy(Cut* newParent = nullptr) const override =0;
	virtual void searchInc(LTreeNode* node, bool fast_mode, bool usefillin = false, const Light_placement& place = Light_placement()) override;
	void searchInc(LTreeNode* node, bool fast_mode, const std::vector<Light_placement>& place);
	void add(SchNode* child);
	const sn_vec& getChildren() const;
	const Bitset& getLayers() const;
	virtual bool contains(lid_t layerid) const override;
	sn_ptr newNode(LTreeNode *_node, const Cluster& _c, bool usefillin = false, const Light_placement& place = Light_placement(), bool fast_mode=false,bool base=false, bool calc_noc=true);
	static sn_ptr newNode(LTreeNode *_node, const Cluster& _c, Cut* parent , bool usefillin=false, const Light_placement& place = Light_placement(), bool fast_mode = false, bool base=false, bool calc_noc = true);
	static sn_ptr newNode(LTreeNode *_node, const Cluster& _c, Cut* parent , const std::vector<Light_placement>& place, bool fast_mode = false,bool base=false,bool calc_noc = true);
	virtual ~Cut() override;
	
	lid_t get_layer_num();

	len_t get_num_bgrp() const;

	friend const Cut* LNode::get_lca(const LNode* node1, const LNode* node2);

	virtual void print_struct(std::string pad = "", std::ostream& os = std::cout) const override;

	virtual const LNode* get_lnode_by_id(lid_t id) const override;
};

class TCut : public Cut{
	/*struct DPParam{
		lid_t last;
	};*/
	virtual void construct(LTreeNode* node, bool usefillin = false, const Light_placement& place = Light_placement()) override;
	virtual void construct_fast(LTreeNode* node) override;
public:
	//TCut(const Bitset& _layers, len_t _bgrp, const Cluster& _c, csn_ptr _parent);
	TCut(LTreeNode* _node, const Cluster& _c, cut_ptr _parent, const std::vector<Light_placement>& place, bool fast_mode = false,bool base=false, bool calc_noc = true);
	TCut(LTreeNode* _node, const Cluster& _c, cut_ptr _parent, bool usefillin = false, const Light_placement& place = Light_placement(), bool fast_mode = false, bool base=false, bool calc_noc = true);
	//virtual SchCost search(lid_t depth) override;
	virtual SchNode* copy(Cut* newParent = nullptr) const override;
	virtual ~TCut() override = default;
	virtual void add_workload_and_dfs(len_t batch_offset, len_t segment, std::vector<Json::Value>& workload_list) const override;
};

class SCut : public Cut{
	//void search(lid_t depth, len_t c_bgrp, SchCost& last_cost);
	virtual void construct(LTreeNode* node, bool usefillin = false, const Light_placement& place = Light_placement()) override;
	virtual void construct_fast(LTreeNode* node) override;
	const std::vector<lid_t> stage;
	const lid_t num_stage;
public:
	//SCut(const Bitset& _layers, len_t _bgrp, const Cluster& _c, csn_ptr _parent);
	SCut(LTreeNode* _node, const Cluster& _c, cut_ptr _parent, bool usefillin = false, const Light_placement& place = Light_placement(),  bool fast_mode = false,bool base=false, bool calc_noc = true);
	//virtual SchCost search(lid_t depth) override;
	virtual SchNode* copy(Cut* newParent = nullptr) const override;
	virtual ~SCut() override = default;
	virtual void add_workload_and_dfs(len_t batch_offset, len_t segment, std::vector<Json::Value>& workload_list) const override;
};

typedef SchNode::sn_ptr sn_ptr;
typedef SchNode::sn_ref sn_ref;

#endif // SCHNODE_H
