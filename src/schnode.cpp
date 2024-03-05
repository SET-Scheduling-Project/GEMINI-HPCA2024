#include "schnode.h"

#include <cassert>
#include <iostream>
#include <math.h>
#include "spatial_mapping/segmentation.h"
#include "layerengine.h"
#include "noc.h"
#include "placement.h"
#include "util.h"
#include "json/json.h"

//sn_ptr SchNode::root=nullptr;
LayerEngine* SchNode::layerMapper =nullptr;
LayerEngine* SchNode::layerMapper_fast = nullptr;
len_t SchNode::tot_batch=0;
SchNode::csn_ptr SchNode::root;
SchNode::wlid_t SchNode::workload_cnt;
SchNode::tfid_t SchNode::transferid_cnt;
std::vector<std::vector<std::vector<SchNode::jsonindex_t> > > SchNode::wlid;
std::vector<bool> SchNode::from_core, SchNode::weight_from_core, SchNode::to_dram;
std::vector<std::map<fmap_range, SchNode::jsonindex_t> > SchNode::ofmapid;
std::vector<std::set<Json::Value> > SchNode::curr_ifmap, SchNode::curr_weight;
std::map<std::string,lid_t> SchNode::name_to_id;
Json::Value SchNode::DRAM;
std::map<Json::Value,SchNode::jsonindex_t> SchNode::DRAM_ofmap_pos;
std::map<Json::Value,SchNode::jsonindex_t> SchNode::DRAM_weight_pos;
std::map<SchNode::tfid_t,SchNode::jsonindex_t> SchNode::DRAM_ifmap_pos;
bw_t SchNode::DRAM_bw;
vol_t SchNode::ubuf;
bool SchNode::cluster_base;
SchNode::energy_breakdown SchNode::record;
/*
sn_ptr SchNode::newNode(SchNode::NodeType t, const Bitset& layers, len_t _bgrp, const Cluster& _c){
	if(layers.count() == 1){
		return new LNode(layers.first(), _bgrp, _c, this);
	}else if(t == SchNode::NodeType::S){
		return new SCut(layers, _bgrp, _c, this);
	}else{
		return new TCut(layers, _bgrp, _c, this);
	}
}
*/

void init(Light_placement& place, SchNode* root){
	assert(root->get_type()==SchNode::NodeType::S);
    auto pt = dynamic_cast<SCut*>(root);
	place.placement.resize(root->getCluster().num_cores());
	place.partition.resize(pt->get_layer_num());
	place.layer_num = pt->get_layer_num();
	lid_t childno=0;
	for(auto child : pt->getChildren()){
		auto lnp = dynamic_cast<LNode*>(child);
		const Node& layerT = lnp->getLayer();
		auto layer_id = layerT.getid();
		place.partition[childno].siz = lnp->getCluster().num_cores();
		place.partition[childno].init(layerT.layer().ofmap_shape(), lnp->get_num_batch(), layer_id);
		place.partition[childno].layerno= layer_id;
		place.fetch_scheme[layer_id] = lnp->get_place_sch().fetch;
		Cluster c = lnp->getCluster();
		assert(!place.layer_DRAM.count(layer_id));
		place.layer_DRAM[layer_id].resize(3);
		place.layer_DRAM[layer_id][0] = layerT.get_external_C() == 0 ? -1 : -2;//lnp->get_nearest_dram();
		place.layer_DRAM[layer_id][1] = layerT.hasWgtPrevs() ? -1 : -2;//lnp->get_nearest_dram();
		place.layer_DRAM[layer_id][2] = lnp->get_to_dram() ? -2:-1;
		c.set_core_set();
		place.layer_scheme[layer_id] = c.core_set;
		place.layers.set(layer_id);
		auto &sch = lnp->get_place_sch();
		place.partition[childno].b = sch.part.B;
		place.partition[childno].c = sch.part.K;
		place.partition[childno].h = sch.part.H;
		place.partition[childno].w = sch.part.W;

		std::map<lid_t,int> vis[700];
		
		for(auto layout: sch.getOfmL()){
			auto order=lnp->get_place_sch().order;
			auto &part=lnp->get_place_sch().part;
			auto shape = layout.first;
			auto core = layout.second;
			auto cluster = pt->getCluster();
			cidx_t cid = -1;
			for(cidx_t i=0;i<Cluster::xlen*Cluster::ylen;++i){
				if(Cluster::get_pos(i)==core){
					cid=i;
					break;
				}
			}
			place.placement[cid].layerno=lnp->getLayer().getid();
			for(int no=0;no<cluster.num_cores();++no){
				if(shape==range_from_partition_number(lnp->getLayer().layer().ofmap_shape(), lnp->get_num_batch(), part, no)){
					place.placement[cid].partno=no;
					assert(!vis[lnp->getLayer().getid()].count(no));
					vis[cid][no]=1;
					break;
				}
			}
		}
		++childno;
	}
}

sn_ptr Cut::newNode(LTreeNode *_node, const Cluster& _c, Cut* parent, const std::vector<Light_placement>& place, bool fast_mode, bool base, bool calc_noc){
	assert(_node->get_type()==NodeType::T);
	return new TCut(_node, _c, parent, place, fast_mode,base,calc_noc);
}

sn_ptr Cut::newNode(LTreeNode* _node, const Cluster& _c, Cut* parent, bool usefillin, const Light_placement& place, bool fast_mode, bool base,bool calc_noc){
	switch (_node->get_type()) {
		case NodeType::L:
			if(usefillin)
				return new LNode(_node, _c, parent, place, fast_mode,base,calc_noc);
			return new LNode(_node, _c, parent,fast_mode,base,calc_noc);
		case NodeType::S:
			return new SCut(_node, _c, parent, usefillin, place, fast_mode,base,calc_noc);
		case NodeType::T:
			return new TCut(_node, _c, parent, usefillin, place, fast_mode,base,calc_noc);
	}
	assert(false);
	return nullptr;
}

len_t Cut::get_num_bgrp() const{
	return num_bgrp;
}

energy_t SchNode::get_ubuf_energy() const{
	return ubuf_energy;
}

energy_t SchNode::get_buf_energy() const{
	return buf_energy;
}

energy_t SchNode::get_bus_energy() const{
	return bus_energy;
}

energy_t SchNode::get_mac_energy() const{
	return mac_energy;
}
access_t SchNode::get_dram_access() const {
	return dram_access;
}

void SchNode::write_energy_record() const {
	record.ubuf = ubuf_energy;
	record.buf = buf_energy;
	record.bus = bus_energy;
	record.mac = mac_energy;
	record.NoC_hop_cost = noc.get_hop_cost();
	record.NoP_hop_cost = noc.get_NoP_hop_cost();
	record.DRAM_cost = noc.get_tot_DRAM_cost();
}

SchNode::SchNode(NodeType t, const Cluster& _c, cut_ptr _parent, len_t nbatch)
	:valid(true), type(t), num_batch(nbatch),
	wgt_bgrp(_parent != nullptr ? _parent->wgt_bgrp : LNode::tot_batch),
	cluster(_c), parent(_parent), //cur_vec(nullptr),
	lnodeList(parent != nullptr ? parent->lnodeList : new nodeList_t) {
	assert(nbatch == 0 || _parent == nullptr || _parent->num_batch % nbatch == 0);
	if (_parent != nullptr) _parent->add(this);
}

void SchNode::setParent(Cut* newParent){
	const_cast<Cut*&>(parent) = newParent;
	if(newParent == nullptr){
		const_cast<nodeList_t*&>(lnodeList) = new nodeList_t;
	}else{
		const_cast<nodeList_t*&>(lnodeList) = newParent->lnodeList;
		newParent->add(this);
	}
}

SchNode::~SchNode(){
	if(parent == nullptr) delete lnodeList;
	// lnodeList = nullptr;
}

bool SchNode::is_valid() const{
	return valid;
}

SchNode::SchCost SchNode::get_cost() const{
	return cost;
}

SchNode::NodeType SchNode::get_type() const{
	return type;
}

const NoC& SchNode::get_noc() const{
	return noc;
}

const Cluster& SchNode::getCluster() const{
	return cluster;
}

const BufferUsage& SchNode::get_buf_usage() const{
	return buf_usage;
}

const BufferUsage& SchNode::get_ifm_usage() const{
	return ifm_usage;
}

const BufferUsage& SchNode::get_wgt_usage() const{
	return wgt_usage;
}

const vol_t SchNode::get_ifm() const {
	return ifm_buf;
}

const vol_t SchNode::get_wgt() const {
	return wgt_buf;
}


const vol_t SchNode::get_fmap() const {
	return fmap_buf;
}

const vol_t SchNode::get_core_mul_data() const {
	return core_mul_data;
}

const vol_t SchNode::get_data() const {
	return tot_data;
}

const len_t SchNode::get_num_batch() const{
	return num_batch;
}


bool SchNode::is_DRAM_cut() const{
	return parent == nullptr && type == NodeType::T;
}

void SchNode::print_res(std::ostream& os) const{
	os << cost << ", Ubuf/Buf/Bus/Mac/NoC/DRAM:" << ubuf_energy << '/' << buf_energy << '/' << bus_energy << '/' << mac_energy;
	os << '/' << noc.get_hop_cost() << '/' << noc.get_cost() - noc.get_hop_cost();
	energy_t e = cost.energy;
	e -= ubuf_energy + buf_energy + bus_energy + mac_energy + noc.get_cost();
	if(e == 0 && cost.energy == 0) return;
	if(cost.energy != 0) e /= cost.energy;
	if(e>1e-8 || e<-1e-8){
		os << std::endl << "[Error]: cost mismatch! error: " << e;
	}
}

std::ostream& operator<<(std::ostream& os, const SchNode& sch){
	sch.print_res(os);
	return os;
}

std::ostream& operator<<(std::ostream& os, const SchNode* sch){
	sch->print_res(os);
	return os;
}

/*
SchNode::SchNode(NodeType t, len_t _bgrp, const Cluster& _c)
	:SchNode(t, _bgrp, _c, this){
	assert(tot_batch % _bgrp == 0);
}
*/
/*
SchNode* SchNode::search_all(lid_t from, lid_t to, len_t bgrp){
// Input: cluster
//
}
*/

bool LNode::search(bool usefillin, const Light_placement &place){
	auto res = usefillin? layerMapper->fillin(this, place,calc_noc,is_base) : layerMapper->search(this, calc_noc);
	if(!res.isValid()) return false;
	noc = std::move(res.noc);
	noc.calc_bw = NoC::calc_noc_control;
	noc.is_base = NoC::interleave;
	ubuf_energy = res.extUbufEnergy;
	place_sch = std::move(res.place);
	tileSch = res.tileSch;
	cost = res.totCost;
	return true;
}

bool LNode::search_fast() {
	auto res = layerMapper_fast->search(this);
	if (!res.isValid()) return false;
	dram_access = res.dram_access;
	wgt_buf = res.weight_buf;
	ifm_buf = res.ifmap_buf;
	tot_data = LNode::tot_batch / num_batch * (wgt_buf + ifm_buf);
	fmap_buf = ifm_buf + ofm_ubuf_vol;
	cost = res.totCost;
	cost.time /= cluster.num_cores();
	return true;
}

/*LNode::LNode(lid_t _layerid, const Cluster& _c, cut_ptr _parent, len_t nbatch)
	:SchNode(NodeType::L, _c, _parent, nbatch), layerid(_layerid),
	  layert(network->getNode(_layerid)), place_sch(_c, layert, nbatch),
	dirp_set(), to_dram(){
	search();
	if(lnodes.size() > layerid){
		lnodes[layerid]=this;
	}else if(lnodes.size() == layerid){
		lnodes.push_back(this);
	}else{
		lnodes.resize(layerid+1, nullptr);
		lnodes.back()=this;
	}
}*/

LNode::LNode(LTreeNode* _node, const Cluster& _c, cut_ptr _parent, const Light_placement& place, bool fast_mode, bool base, bool calc_noc_)
	:SchNode(NodeType::L, _c, _parent, _node->get_tot_batch()), layerid(_node->layers().first()),
	  layert(network->getNode(layerid)), /*place_sch(cluster, layert, _node->get_tot_batch()),*/
	dirp_set(_node->get_dirp_set()), to_dram(_node->get_to_dram()){
	is_base = base;
	calc_noc = NoC::calc_noc_control;
	Cluster c_temp = cluster;
	nearest_dram_id = c_temp.nearest_dram();
	if (!fast_mode) {
		searchLayer(true,place);
	}
	else {
		searchLayer_fast();
	}
}

LNode::LNode(LTreeNode *_node, const Cluster& _c, SchNode::cut_ptr _parent, bool fast_mode, bool base, bool calc_noc_)
	:SchNode(NodeType::L, _c, _parent, _node->get_tot_batch()), layerid(_node->layers().first()),
	  layert(network->getNode(layerid)), /*place_sch(cluster, layert, _node->get_tot_batch()),*/
	dirp_set(_node->get_dirp_set()), to_dram(_node->get_to_dram()){
	/*
	if (layerid == 79) {
		std::cout << "\n";
	}*/
	is_base = base;
	calc_noc = NoC::calc_noc_control;
	Cluster c_temp = cluster;
	nearest_dram_id = c_temp.nearest_dram();
	if (!fast_mode) {
		searchLayer();
	}
	else {
		searchLayer_fast();
	}
}

SchNode* LNode::copy(Cut* newParent) const{
	LNode* node = new LNode(*this);
	node->setParent(newParent);
	(*node->lnodeList)[layerid] = node;
	return node;
}

void LNode::searchInc(LTreeNode* node, bool fast_mode=false, bool usefillin, const Light_placement& place){
	(void) node;
	// if(!node->isModified()) return;
	// We'll never use this function.
	// Since LNode can't be child-modified.
	assert(false);
	return;
}

void LNode::searchLayer(bool usefillin, const Light_placement &place){
	//std::cout << "Starting " << layert.layer().get_name() << std::endl;
	valid = search(usefillin, place);
	/*
	std::cout << place_sch.part;
	for(int i=0;i<4;++i){
		std::cout << ' ' << (int)place_sch.order[i];
	}
	std::cout << ' ' << cost;
	std::cout << std::endl;
	*/
	//std::cout << cost.time << " v.s. " << noc.get_time() << std::endl;
	if(!valid){
		return;
	}
	if(!place_sch.getIfmL().update(ifm_usage, layert.layer(), place_sch.fetch, true)){
		valid = false;
		return;
	}
	if(!place_sch.getWgtL().update(layert.hasWgtPrevs() ? ifm_usage : wgt_usage, layert.layer(), place_sch.fetch, false)){
		valid = false;
		return;
	}

	buf_usage = ifm_usage;
	if(!buf_usage.all_add(ofm_ubuf_vol)){
		valid = false;
		return;
	}
	if(!(buf_usage + wgt_usage)){
		valid = false;
		return;
	}
	(*lnodeList)[layerid] = this;
	ubuf_energy += tileSch.ubuf * cluster.num_cores();
	buf_energy = tileSch.buffer * cluster.num_cores();
	bus_energy = tileSch.noc * cluster.num_cores();
	mac_energy = tileSch.mac * cluster.num_cores();
	bool is_seg = (parent == nullptr) || parent->is_DRAM_cut();
	if(is_seg){
		cycle_t noc_time = noc.get_time();
		cost.time = MAX(cost.time, noc_time);
	}
}

void LNode::searchLayer_fast() {
	//std::cout << "Starting " << layert.layer().get_name() << std::endl;
	ifm_buf = wgt_buf = fmap_buf = 0;
	tot_data = core_mul_data = 0;
	valid = search_fast();
	
	/*
	std::cout << place_sch.part;
	for(int i=0;i<4;++i){
		std::cout << ' ' << (int)place_sch.order[i];
	}
	std::cout << ' ' << cost;
	std::cout << std::endl;
	*/
	//std::cout << cost.time << " v.s. " << noc.get_time() << std::endl;
	if (!valid) {
		return;
	}
	/*
	if (!place_sch.getIfmL().update(ifm_usage)) {
		valid = false;
		return;
	}
	if (!place_sch.getWgtL().update(layert.hasWgtPrevs() ? ifm_usage : wgt_usage)) {
		valid = false;
		return;
	}

	buf_usage = ifm_usage;
	if (!buf_usage.all_add(ofm_ubuf_vol)) {
		valid = false;
		return;
	}
	if (!(buf_usage + wgt_usage)) {
		valid = false;
		return;
	}*/
	if (parent->get_type() == NodeType::T) {
		if (fmap_buf + wgt_buf + ofm_ubuf_vol > ubuf) {
			valid = false;
			return;
		}
	}
	else if (parent->get_type() == NodeType::S) {
			//if (num_batch == tot_batch) {
				if (fmap_buf + wgt_buf + ofm_ubuf_vol > ubuf) {
					valid = false;
					return;
				}
			//}
				/*
			else {
				if (fmap_buf * 2 + wgt_buf + ofm_ubuf_vol > ubuf) {
					valid = false;
					return;
				}
			}*/
	}
	
	(*lnodeList)[layerid] = this;
	/*
	ubuf_energy += tileSch.ubuf * cluster.num_cores();
	buf_energy = tileSch.buffer * cluster.num_cores();
	bus_energy = tileSch.noc * cluster.num_cores();
	mac_energy = tileSch.mac * cluster.num_cores();*/
	bool is_seg = (parent == nullptr) || parent->is_DRAM_cut();
	if (is_seg) {
		cost.time = MAX(cost.time, DIVCEIL(dram_access, (4 * DRAM_bw)));
	}
}

/*
LNode::LNode(const Bitset& _layers, len_t _bgrp, const Cluster& _c, SchNode::csn_ptr _parent)
	:SchNode(NodeType::L, _bgrp, _c, _parent),
	  layert(network->getNode(_layers.first())){
	assert(_layers.count() == 1);

}
*/

bool LNode::contains(lid_t _layerid) const{
	return _layerid == layerid;
}

const Node& LNode::getLayer() const{
	return layert;
}

const PlaceSch& LNode::get_place_sch() const{
	return place_sch;
}

const Bitset& LNode::get_dirp_set() const{
	return dirp_set;
}

mlen_t LNode::get_nearest_dram() const {
	return nearest_dram_id;
}

bool LNode::get_to_dram() const{
	return to_dram;
}

LNode::~LNode(){
	(*lnodeList)[layerid] = nullptr;
}
bool LNode::is_seg() const {
	return parent->is_DRAM_cut()||(parent->get_type()==NodeType::S&&parent->getLayers().count()==1);
}
const Cut* LNode::get_lca(const LNode* node1, const LNode* node2){
	const Cut* lca = node2->parent;
	while (!lca->layers.contains(node1->layerid)) {
		lca = lca->parent;
	}
	return lca;
}

void LNode::print_struct(std::string pad, std::ostream& os) const{
	os << pad << layert.name() << ' ' << num_batch << ' ' << place_sch;
	os << " util:" << tileSch.util*100 << '/' << tileSch.tot_util*100;
	os << ' ' << cost << " Ubuf/Buf/Bus/Mac:" << ubuf_energy << '/' << buf_energy << '/' << bus_energy << '/' << mac_energy;
	os << " NoC energy = " << noc.get_hop_cost() << " DRAM energy = " << noc.get_tot_DRAM_cost();
	os << ' ' << noc; //<< ' ' << buf_usage << ' ' << wgt_usage << ' ' << ifm_usage;
	os << ' ' << layert.layer().real_ifmap_shape().tot_size(num_batch);
	os << '/' << layert.layer().weight_size();
	os << '/' << layert.layer().ofmap_shape().tot_size(num_batch);
	
	os << " NoCtime = " << noc.get_time() << " NocBW = " << NoC::NoC_bw << " Computing_time = " << tileSch.cost.time;
	os << "DDR_access 0 1 2 3 =" << noc.get_DRAM_acc(0) << "," << noc.get_DRAM_acc(1) << "," << noc.get_DRAM_acc(2) << "," << noc.get_DRAM_acc(3) << ",";
	os << std::endl;
	if(parent == nullptr || parent->is_DRAM_cut()){
		os << pad << "NoC info:";
		size_t i = 0;
		for(const auto& it: noc.get_link_info()){
			os << it << ' ';
			if(++i >= 4) break;
		}
		os << std::endl;
	}
}

/*
Cut::Cut(NodeType t, const Bitset& _layers, const Cluster& _c, csn_ptr _parent, len_t nbgrp, len_t nbatch)
	:SchNode(t, _c, _parent, _bgrp), layers(_layers){
	assert(t!=NodeType::L);
	assert(_layers.count() > 1);
}
*/
Cut::Cut(SchNode::NodeType t, LTreeNode* node, const Cluster& _c, SchNode::cut_ptr _parent)
	:SchNode(t, _c, _parent, node->get_tot_batch()), curNode(nullptr),
	  layers(node->layers()), num_bgrp(node->get_bgrp_num()){
}

void Cut::searchInc(LTreeNode* node, bool fast_mode, const std::vector<Light_placement>& place){
	assert(node->layers() == layers);
	curNode = node;
	oldChildren = std::move(children);
	children.clear();
	noc = NoC(!is_base, NoC::interleave);
	ifm_usage = BufferUsage();
	wgt_usage = BufferUsage();
	buf_usage = BufferUsage();
	if (!fast_mode) {
		construct(node, place);
	}
	else {
		construct_fast(node);
	}
	while(!oldChildren.empty()){
		delete oldChildren.front();
		oldChildren.pop_front();
	}
	curNode = nullptr;
}

void Cut::searchInc(LTreeNode* node, bool fast_mode=false, bool usefillin, const Light_placement& place){
	assert(node->layers() == layers);
	curNode = node;
	oldChildren = std::move(children);
	children.clear();
	noc = NoC(!is_base, NoC::interleave);
	ifm_usage = BufferUsage();
	wgt_usage = BufferUsage();
	buf_usage = BufferUsage();
	if (!fast_mode) {
		construct(node, usefillin, place);
	}
	else {
		construct_fast(node);
	}
	while(!oldChildren.empty()){
		delete oldChildren.front();
		oldChildren.pop_front();
	}
	curNode = nullptr;
}

sn_ptr Cut::newNode(LTreeNode* _node, const Cluster& _c, bool usefillin, const Light_placement& place, bool fast_mode,bool base, bool calc_noc_){
	if(curNode == nullptr) return newNode(_node, _c, this,usefillin, place, fast_mode,base,calc_noc);
	if(_node->isNew()) return newNode(_node, _c, this, usefillin, place, fast_mode,base, calc_noc);
	is_base = base;
	calc_noc = NoC::calc_noc_control;
	const Bitset& layers = _node->layers();
	bool found = false, reSearch = false;
	while(!oldChildren.empty()){
		SchNode* node = oldChildren.front();
		oldChildren.pop_front();
		switch(node->get_type()){
		case NodeType::L:
			if(layers.count() == 1 && node->contains(layers.first()))
				found = true;
			break;
		case NodeType::S:
			reSearch = (_c != node->getCluster());
		[[clang::fallthrough]];
		case NodeType::T:
			Cut* cut = static_cast<Cut*>(node);
			if(cut->layers == layers) found = !reSearch;
			break;
		}
		if(found){
			children.push_back(node);
			if(_node->isModified())
				node->searchInc(_node,fast_mode, usefillin, place);
			return node;
		}
		delete node;
		if(reSearch) return newNode(_node, _c, this, usefillin, place, fast_mode,base,calc_noc);
	}
	std::cerr << "[Warning] Cannot find old child in oldChildren." << std::endl;
	return newNode(_node, _c, this, usefillin, place, fast_mode,base,calc_noc);
}

void Cut::add(SchNode* child){
	children.push_back(child);
}

const SchNode::sn_vec& Cut::getChildren() const{
	return children;
}
const Bitset& Cut::getLayers() const {
	return layers;
}
lid_t Cut::get_layer_num(){
	return layers.count();
}

bool Cut::contains(lid_t layerid) const{
	return layers.contains(layerid);
}

Cut::~Cut(){
	for(auto &child: children){
		delete child;
	}
}

void Cut::print_struct(std::string pad, std::ostream& os) const{
	os << pad << ((type == NodeType::S)?'S':'T');
	os << ' ' << num_batch << '/' << num_bgrp;
	os << ' ' << cost << " Ubuf/Buf/Bus/Mac:" << ubuf_energy << '/' << buf_energy << '/' << bus_energy << '/' << mac_energy;
	os << " NoC energy = " << noc.get_NoC_hop_cost() << " NoP energy = " << noc.get_NoP_hop_cost() << " DRAM energy = " << noc.get_tot_DRAM_cost();
	os << ' ' << noc; //<< ' ' << buf_usage << ' ' << wgt_usage << ' ' << ifm_usage;
	os << " NoCtime = " << noc.get_time() << " NocBW = " << NoC::NoC_bw ;
	os << " DDR_access 0 1 2 3 =" << noc.get_DRAM_acc(0) << "," << noc.get_DRAM_acc(1) << "," << noc.get_DRAM_acc(2) << "," << noc.get_DRAM_acc(3) << ",";
	os << std::endl;
	pad += '\t';
	for(auto child: children){
		child->print_struct(pad, os);
	}
	if(type == NodeType::S){
		if(parent == nullptr || parent->is_DRAM_cut()){
			os << pad << "NoC info:";
			size_t i = 0;
			for(const auto& it: noc.get_link_info()){
				os << it << ' ';
				if(++i >= 4) break;
			}
			os << std::endl;
		}
	}else{
		if(parent != nullptr && parent->is_DRAM_cut()){
			os << pad << "NoC info:";
			size_t i = 0;
			for(const auto& it: noc.get_link_info()){
				os << it << ' ';
				if(++i >= 4) break;
			}
			os << std::endl;
		}
	}
}
/*
TCut::TCut(const Bitset& _layers, len_t _bgrp, const Cluster& _c, csn_ptr _parent)
	:Cut(NodeType::T, _layers, _bgrp, _c, _parent){}
*/
TCut::TCut(LTreeNode* _node, const Cluster& _c, cut_ptr _parent, const std::vector<Light_placement>& place, bool fast_mode,bool base,bool calc_noc_)
	:Cut(NodeType::T, _node, _c, _parent){
	is_base = base;
	calc_noc = NoC::calc_noc_control;
	if (!fast_mode) {
		Cut::construct(_node, place);
	}
	else {
		TCut::construct_fast(_node);
	}
}

TCut::TCut(LTreeNode *_node, const Cluster& _c, SchNode::cut_ptr _parent, bool usefillin, const Light_placement& place, bool fast_mode,bool base,bool calc_noc_)
	:Cut(NodeType::T, _node, _c, _parent){
	is_base = base;
	calc_noc = NoC::calc_noc_control;
	if (!fast_mode) {
		TCut::construct(_node, usefillin, place);
	}
	else {
		TCut::construct_fast(_node);
	}
}

SchNode* TCut::copy(Cut* newParent) const{
	TCut* cut = new TCut(*this);
	cut->setParent(newParent);
	cut->children.clear();
	for(auto child : children){
		child->copy(cut);
	}
	return cut;
}

void Cut::construct(LTreeNode* node, const std::vector<Light_placement>& place){
	assert(type==NodeType::T);
	const bool usefillin = true;
	bool is_top = (parent == nullptr);
	bool is_seg = (!is_top) && parent->is_DRAM_cut();
	bool wgt_shift = is_seg && (num_bgrp == 1);
	if (wgt_shift) {
		// If shift weight, we need to fetch weight for each subgrp.
		wgt_bgrp = num_batch / num_bgrp;
	}

	noc.calc_bw = NoC::calc_noc_control;
	noc.is_base = NoC::interleave;
	sn_ptr last_p = nullptr;
	cost.energy = 0;
	cost.time = 0;
	ubuf_energy = buf_energy = bus_energy = mac_energy = 0;
	int childno = 0;
	for(auto child: node->get_children()){
		sn_ptr p = newNode(child, cluster, usefillin, place[childno++], false,is_base,calc_noc);
		if(!p->is_valid()){
			valid = false;
			/*sn_ptr x = p;
			while(dynamic_cast<LNode*>(x)==nullptr){
				Cut* c = dynamic_cast<Cut*>(x);
				x =c->getChildren().back();
			}
			std::cout << dynamic_cast<LNode*>(x)->getLayer().name() << std::endl;
			*/
			return;
		}
		if(!is_top){
			/*if(!p->get_ifm_usage()){
				valid = false;
				return;
			}*/
			if(!(is_seg || (ifm_usage += p->get_ifm_usage()))){
				valid = false;
				return;
			}
			if(last_p == nullptr){
				//buf_usage = p->get_buf_usage();
			}else{
				if(wgt_shift){
					buf_usage.max(p->get_ifm_usage() + last_p->get_buf_usage() + last_p->get_wgt_usage() + p->get_wgt_usage());
				}else{
					buf_usage.max(p->get_ifm_usage() + last_p->get_buf_usage());
				}
				if(!buf_usage){
					valid = false;
					return;
				}
				//buf_usage.max(p->get_buf_usage());
			}
			if(!wgt_shift && !(wgt_usage += p->get_wgt_usage())){
				valid = false;
				return;
			}
		}
		cost.time += p->get_cost().time;
		cost.energy += p->get_cost().energy;
		noc += p->get_noc();
		ubuf_energy += p->get_ubuf_energy();
		buf_energy += p->get_buf_energy();
		bus_energy += p->get_bus_energy();
		mac_energy += p->get_mac_energy();
		last_p = p;
	}
	if(!is_top){
		if(num_bgrp == 1){
			if(wgt_shift){
				buf_usage.max(last_p->get_buf_usage() + last_p->get_wgt_usage());
			}else{
				buf_usage.max(last_p->get_buf_usage());
			}
		}else{
			if(!(is_seg || ifm_usage.multiple(num_bgrp))){
				valid = false;
				return;
			}
			if(wgt_shift){
				buf_usage.max(children.front()->get_ifm_usage() + children.front()->get_wgt_usage() + last_p->get_buf_usage() + last_p->get_wgt_usage());
			}else{
				buf_usage.max(children.front()->get_ifm_usage() + last_p->get_buf_usage());
			}
		}
		if(!buf_usage){
			valid = false;
			return;
		}
		if(!wgt_shift && !(buf_usage + wgt_usage)){
			valid = false;
			return;
		}
	}
	cost *= num_bgrp;
	noc *= num_bgrp;
	ubuf_energy *= num_bgrp;
	buf_energy *= num_bgrp;
	bus_energy *= num_bgrp;
	mac_energy *= num_bgrp;
	// Needs to bound total time with DRAM access.
	if(is_seg){
		cycle_t noc_time = noc.get_time();
		cost.time = MAX(cost.time, noc_time);
	}
}


void TCut::construct(LTreeNode* node, bool usefillin, const Light_placement& place){
	bool is_top = (parent == nullptr);
	bool is_seg = (!is_top) && parent->is_DRAM_cut();
	bool wgt_shift = is_seg && (num_bgrp == 1);
	if (wgt_shift) {
		// If shift weight, we need to fetch weight for each subgrp.
		wgt_bgrp = num_batch / num_bgrp;
	}
	sn_ptr last_p = nullptr;
	cost.energy = 0;
	cost.time = 0;
	noc.calc_bw = NoC::calc_noc_control;
	noc.is_base = NoC::interleave;
	ubuf_energy = buf_energy = bus_energy = mac_energy = 0;
	for(auto child: node->get_children()){
		if (child->layers().contains(58)) {
			std::cout << "\n";
		}
		sn_ptr p = newNode(child, cluster, usefillin, place, false, is_base,calc_noc);
		if(!p->is_valid()){
			valid = false;
			/*sn_ptr x = p;
			while(dynamic_cast<LNode*>(x)==nullptr){
				Cut* c = dynamic_cast<Cut*>(x);
				x =c->getChildren().back();
			}
			std::cout << dynamic_cast<LNode*>(x)->getLayer().name() << std::endl;
			*/
			return;
		}
		if(!is_top){
			/*if(!p->get_ifm_usage()){
				valid = false;
				return;
			}*/
			if(!(is_seg || (ifm_usage += p->get_ifm_usage()))){
				valid = false;
				return;
			}
			if(last_p == nullptr){
				//buf_usage = p->get_buf_usage();
			}else{
				if(wgt_shift){
					buf_usage.max(p->get_ifm_usage() + last_p->get_buf_usage() + last_p->get_wgt_usage() + p->get_wgt_usage());
				}else{
					buf_usage.max(p->get_ifm_usage() + last_p->get_buf_usage());
				}
				if(!buf_usage){
					valid = false;
					return;
				}
				//buf_usage.max(p->get_buf_usage());
			}
			if(!wgt_shift && !(wgt_usage += p->get_wgt_usage())){
				valid = false;
				return;
			}
		}
		cost.time += p->get_cost().time;
		cost.energy += p->get_cost().energy;
		noc += p->get_noc();
		ubuf_energy += p->get_ubuf_energy();
		buf_energy += p->get_buf_energy();
		bus_energy += p->get_bus_energy();
		mac_energy += p->get_mac_energy();
		last_p = p;
	}
	if(!is_top){
		if(num_bgrp == 1){
			if(wgt_shift){
				buf_usage.max(last_p->get_buf_usage() + last_p->get_wgt_usage());
			}else{
				buf_usage.max(last_p->get_buf_usage());
			}
		}else{
			if(!(is_seg || ifm_usage.multiple(num_bgrp))){
				valid = false;
				return;
			}
			if(wgt_shift){
				buf_usage.max(children.front()->get_ifm_usage() + children.front()->get_wgt_usage() + last_p->get_buf_usage() + last_p->get_wgt_usage());
			}else{
				buf_usage.max(children.front()->get_ifm_usage() + last_p->get_buf_usage());
			}
		}
		if(!buf_usage){
			valid = false;
			return;
		}
		if(!wgt_shift && !(buf_usage + wgt_usage)){
			valid = false;
			return;
		}
	}
	cost *= num_bgrp;
	noc *= num_bgrp;
	ubuf_energy *= num_bgrp;
	buf_energy *= num_bgrp;
	bus_energy *= num_bgrp;
	mac_energy *= num_bgrp;
	// Needs to bound total time with DRAM access.
	if(is_seg){
		cycle_t noc_time = noc.get_time();
		cost.time = MAX(cost.time, noc_time);
	}
}

void TCut::construct_fast(LTreeNode* node) {
	if (node->layers()[27]) {
//		std::cout << "1";
	}
	bool is_top = (parent == nullptr);
	bool is_seg = (!is_top) && parent->is_DRAM_cut();
	bool wgt_shift = is_seg && (num_bgrp == 1);
	if (wgt_shift) {
		// If shift weight, we need to fetch weight for each subgrp.
		wgt_bgrp = num_batch / num_bgrp;
	}
	sn_ptr last_p = nullptr;
	cost.energy = 0;
	cost.time = 0;
	dram_access = 0;
	ubuf_energy = buf_energy = bus_energy = mac_energy = 0;
	ifm_buf = wgt_buf = fmap_buf = 0;
	tot_data = core_mul_data = 0;
	//vol_t min_buffer_tmp = ubuf;
	for (auto child : node->get_children()) {
		sn_ptr p = newNode(child, cluster,false,Light_placement(),true);
		tot_data += p->get_data();
		core_mul_data += p->get_core_mul_data();
		if (!p->is_valid()) {
			valid = false;
			/*sn_ptr x = p;
			while(dynamic_cast<LNode*>(x)==nullptr){
				Cut* c = dynamic_cast<Cut*>(x);
				x =c->getChildren().back();
			}
			std::cout << dynamic_cast<LNode*>(x)->getLayer().name() << std::endl;
			*/
			return;
		}
		
		if (!is_top) {
			
			if(!(p->get_ifm()<= ubuf)){
				valid = false;
				return;
			}
			if (!(is_seg || ((ifm_buf += p->get_ifm())<=ubuf))) {
				valid = false;
				return;
			}
			if (last_p == nullptr) {
				//buf_usage = p->get_buf_usage();
			}
			else {
				if (wgt_shift) {
					vol_t tmp = p->get_ifm() + last_p->get_fmap() + last_p->get_wgt() + p->get_wgt();
					if (tmp >= fmap_buf) {
						fmap_buf = tmp;
					}	
				}
				else {
					vol_t tmp = p->get_ifm() + last_p->get_fmap();
					if (tmp >= fmap_buf) {
						fmap_buf = tmp;
					}
				}
				if (fmap_buf>ubuf) {
					valid = false;
					return;
				}
				//buf_usage.max(p->get_buf_usage());
			}
			if (!wgt_shift && !((wgt_buf += p->get_wgt())<=ubuf)) {
				valid = false;
				return;
			}
		}
		cost.time += p->get_cost().time;
		dram_access += p->get_dram_access();
		last_p = p;
		/*
		cost.energy += p->get_cost().energy;
		noc += p->get_noc();
		ubuf_energy += p->get_ubuf_energy();
		buf_energy += p->get_buf_energy();
		bus_energy += p->get_bus_energy();
		mac_energy += p->get_mac_energy();*/
		
		
	}
	dram_access *= num_bgrp;
	if (!is_top) {
		if (num_bgrp == 1) {
			if (wgt_shift) {
				//view weight as ifmap
				vol_t tmp = last_p->get_fmap() + last_p->get_wgt();
				if (tmp > fmap_buf) {
					fmap_buf = tmp;
				}
			}
			else {
				vol_t tmp = last_p->get_fmap();
				if (tmp > fmap_buf) {
					fmap_buf = tmp;
				}
			}
		}
		else {
			if (!(is_seg || (ifm_buf*=num_bgrp)<=ubuf)) {
				valid = false;
				return;
			}
			if (wgt_shift) {
				vol_t tmp = children.front()->get_ifm() + last_p->get_fmap() + last_p->get_wgt() + children.front()->get_wgt();
				if (tmp > fmap_buf) {
					fmap_buf = tmp;
				}
			}
			else {
				vol_t tmp = children.front()->get_ifm() + last_p->get_fmap();
				if (tmp > fmap_buf) {
					fmap_buf = tmp;
				}
			}
		}
		if (fmap_buf>ubuf) {
			valid = false;
			return;
		}
		if (!wgt_shift && !((fmap_buf + wgt_buf)<=ubuf)) {
			valid = false;
			return;
		}
	}
	/*
	cost *= num_bgrp;
	noc *= num_bgrp;
	ubuf_energy *= num_bgrp;
	buf_energy *= num_bgrp;
	bus_energy *= num_bgrp;
	mac_energy *= num_bgrp;
	// Needs to bound total time with DRAM access.
	*/
	if (is_seg) {
		//cycle_t noc_time = noc.get_time();
		cost.time = MAX(cost.time, DIVCEIL(dram_access, (4 * DRAM_bw)));
	}
}
/*
SCut::SCut(const Bitset& _layers, len_t _bgrp, const Cluster& _c, csn_ptr _parent)
	:Cut(NodeType::S, _layers, _bgrp, _c, _parent){}
*/
SCut::SCut(LTreeNode *_node, const Cluster& _c, SchNode::cut_ptr _parent, bool usefillin, const Light_placement& place, bool fast_mode,bool base,bool calc_noc_)
	:Cut(NodeType::S, _node, _c, _parent),
	 stage(_node->get_stages()), num_stage(_node->get_num_stage()){
	is_base = base;
	calc_noc = NoC::calc_noc_control;
	if (!fast_mode) {
		SCut::construct(_node, usefillin, place);
	}
	else {
		SCut::construct_fast(_node);
	}
}

SchNode* SCut::copy(Cut* newParent) const{
	SCut* cut = new SCut(*this);
	cut->setParent(newParent);
	cut->children.clear();
	for(auto child : children){
		child->copy(cut);
	}
	return cut;
}

void SCut::construct_fast(LTreeNode* node){
	if (node->layers()[27]) {
//		std::cout << "1";
	}
	bool is_seg = (parent == nullptr) || parent->is_DRAM_cut();
	const auto& cnodes = node->get_children();
	cidx_t cnum = static_cast<cidx_t>(cnodes.size());
	assert(cnum > 0);
	utime_t* tlist = new utime_t[cnum];
	utime_t* cur_item = tlist;
	for(auto child: cnodes){
		*(cur_item++) = child->get_utime();
	}
	auto allocRes = cluster.try_alloc(tlist, cnum,0.0,cluster_base);
	delete[] tlist;
	if(!allocRes){
		valid = false;
		return;
	}
	cidx_t i=0;
	cycle_t max_time = 0;
	cost.energy = 0;
	dram_access = 0;
	ubuf_energy = buf_energy = bus_energy = mac_energy = 0;
	core_mul_data =tot_data = 0;
	for(auto child: cnodes){
		auto p = newNode(child, cluster.sub_cluster(i++, allocRes),false,Light_placement(),true);
		core_mul_data += p->get_data() * p->getCluster().num_cores();
		if(!p->is_valid()){
			valid = false;
			return;
		}
		fmap_buf = 0;
		if(num_bgrp > 1 && !((fmap_buf += (p->get_ifm()+p->get_fmap()))<=ubuf)){
			valid = false;
			return;
		}
		/*if(static_cast<LNode*>(p)->getLayer().name() == "conv2_0_b"){
			std::cout << "BU:" << std::endl;
			for(auto x: p->get_buf_usage().usage){
				std::cout << (int)x.first.x << ' ' << (int)x.first.y << ' ' << x.second << std::endl;
			}
			std::cout << "CU:" << std::endl;
			for(auto x: buf_usage.usage){
				std::cout << (int)x.first.x << ' ' << (int)x.first.y << ' ' << x.second << std::endl;
			}
		}
		
		if(!(buf_usage += p->get_buf_usage())){
			valid = false;
			return;
		}*/
		/*
		if(!(wgt_usage += p->get_wgt_usage())){
			valid = false;
			return;

		}
		if(!(is_seg || (ifm_usage += p->get_ifm_usage()))){
			valid = false;
			return;
		}
		//cost.time += p->get_cost().time;
		cost.energy += p->get_cost().energy;
		max_time = MAX(p->get_cost().time, max_time);
		noc += p->get_noc();
		ubuf_energy += p->get_ubuf_energy();
		buf_energy += p->get_buf_energy();
		bus_energy += p->get_bus_energy();
		mac_energy += p->get_mac_energy();*/
		max_time = MAX(p->get_cost().time, max_time);
		dram_access += p->get_dram_access();
	}
	/*
	if(!(buf_usage + wgt_usage)){
		valid = false;
		return;
	}
	ifm_usage.multiple(num_bgrp);
	if(!(is_seg || ifm_usage)){
		valid = false;
		return;
	}
	cost.time = max_time * (num_stage + num_bgrp);
	cost.energy *= num_bgrp;
	noc *= num_bgrp;
	ubuf_energy *= num_bgrp;
	buf_energy *= num_bgrp;
	bus_energy *= num_bgrp;
	mac_energy *= num_bgrp;*/
	cost.time = max_time * (num_stage + num_bgrp);
	dram_access *= num_bgrp;
	// Needs to bound total time with DRAM access.
	
	if(is_seg){
		//cycle_t noc_time = noc.get_time();
		cost.time = MAX(cost.time, DIVCEIL(dram_access, (4 * DRAM_bw)));
	}
}
void SCut::construct(LTreeNode* node, bool usefillin, const Light_placement& place) {
	//for debug
	bool is_seg = (parent == nullptr) || parent->is_DRAM_cut();
	const auto& cnodes = node->get_children();
	cidx_t cnum = static_cast<cidx_t>(cnodes.size());
	Cluster::allocRes_t allocRes;
	noc.calc_bw = calc_noc;
	noc.is_base = NoC::interleave;
	assert(cnum > 0);
	if (!usefillin) {
		utime_t* tlist = new utime_t[cnum];
		utime_t* cur_item = tlist;
		for (auto child : cnodes) {
			*(cur_item++) = child->get_utime();
		}
		allocRes = cluster.try_alloc(tlist, cnum,0,cluster_base);
		delete[] tlist;
		if (!allocRes) {
			valid = false;
			return;
		}
	}
	cidx_t i = 0;
	cycle_t max_time = 0;
	cost.energy = 0;
	ubuf_energy = buf_energy = bus_energy = mac_energy = 0;
	for (auto child : cnodes) {
		sn_ptr p;
		if(usefillin){
			std::vector<cidx_t> ids;
			cidx_t cid = 0;
			for(auto core:place.get_placement()){
				if(child->layers().contains(core.layerno)){
					ids.push_back(cid);
				}
				++cid;
			}
			Cluster sub(ids);
			p = newNode(child, sub, usefillin, place, false, is_base,calc_noc);
		}
		else{
			p = newNode(child, cluster.sub_cluster(i++, allocRes), usefillin, place, false, is_base,calc_noc);
		}
		if (!p->is_valid()) {
			valid = false;
			return;
		}
		if (num_bgrp > 1 && !(buf_usage += p->get_ifm_usage())) {
			valid = false;
			return;
		}
		/*if(static_cast<LNode*>(p)->getLayer().name() == "conv2_0_b"){
			std::cout << "BU:" << std::endl;
			for(auto x: p->get_buf_usage().usage){
				std::cout << (int)x.first.x << ' ' << (int)x.first.y << ' ' << x.second << std::endl;
			}
			std::cout << "CU:" << std::endl;
			for(auto x: buf_usage.usage){
				std::cout << (int)x.first.x << ' ' << (int)x.first.y << ' ' << x.second << std::endl;
			}
		}*/
		if (!(buf_usage += p->get_buf_usage())) {
			valid = false;
			return;
		}
		if (!(wgt_usage += p->get_wgt_usage())) {
			valid = false;
			return;

		}
		if (!(is_seg || (ifm_usage += p->get_ifm_usage()))) {
			valid = false;
			return;
		}
		//cost.time += p->get_cost().time;
		cost.energy += p->get_cost().energy;
		max_time = MAX(p->get_cost().time, max_time);
		noc += p->get_noc();
		ubuf_energy += p->get_ubuf_energy();
		buf_energy += p->get_buf_energy();
		bus_energy += p->get_bus_energy();
		mac_energy += p->get_mac_energy();
	}
	if (!(buf_usage + wgt_usage)) {
		valid = false;
		return;
	}
	ifm_usage.multiple(num_bgrp);
	if (!(is_seg || ifm_usage)) {
		valid = false;
		return;
	}
	cost.time = max_time * (num_stage + num_bgrp);
	cost.energy *= num_bgrp;
	noc *= num_bgrp;
	ubuf_energy *= num_bgrp;
	buf_energy *= num_bgrp;
	bus_energy *= num_bgrp;
	mac_energy *= num_bgrp;
	// Needs to bound total time with DRAM access.
	if (is_seg) {
		cycle_t noc_time = noc.get_time();
		cost.time = MAX(cost.time, noc_time);
	}
}
/*
LNode::SchCost LNode::search(lid_t depth){
	//printf("\tL\t%d\t%d\t%d\t%d\n",depth,from,to,cluster.num_cores());
	// Search best scheme for each partition.
	// Add the cost of all sequential deps.
	// Return the best scheme.
	(void)depth;
	len_t tot_part = static_cast<len_t>(cluster.num_cores());
	len_t totK = layert.workload().K;
	len_t totB = bgrp;
	double max_util = -1;
	len_t maxK=0, maxB=0;
	len_t kpart, bpart = tot_part;
	SchCost cost;
	part_engine;
	init(100, wl);
	for (Part part = gen_part(finish, cost);){

	}
	struct PlaceInfo{
	Cluster
	Part
	wl
	};
	PlaceInfo last, P

	struct NoCTraffic{
		access_t x[];
	};


	do{
		kpart = tot_part/bpart;
		if(tot_part%bpart == 0){
			double cur_util = DIVCEIL(totK,kpart) * DIVCEIL(totB, bpart);
			cur_util *= kpart*bpart;
			cur_util = (totK*totB)/cur_util;
			if(cur_util >= Cluster::min_util){
				max_util = 2;
				Node::Workload wl = layert.workload();
				wl.N = DIVCEIL(totB, bpart);
				wl.K = DIVCEIL(totK, kpart);
				wl.calc_op();
				CoreMapper::MapCost c = mapper->genMapping(wl).cost;
				if(c.energy < energy_inf)
					c.energy *= tot_part;
				if(c.cost() < cost.cost()){
					cost.energy = c.energy;
					cost.time = c.time;
				}
			}else if(cur_util > max_util){
				max_util = cur_util;
				maxK = kpart;
				maxB = bpart;
			}
		}
		bpart = tot_part/++kpart;
	}while(bpart>0);
	if(max_util < 1.5 && max_util > 0){
		Node::Workload wl = layert.workload();
		wl.N = DIVCEIL(totB, maxB);
		wl.K = DIVCEIL(totK, maxK);
		wl.calc_op();
		CoreMapper::MapCost c = mapper->genMapping(wl).cost;
		if(c.energy < energy_inf)
			c.energy *= tot_part;
		if(c.cost() < cost.cost()){
			cost.energy = c.energy;
			cost.time = c.time;
		}
	}
	if(cost.energy < energy_inf){
		cost.energy *= tot_batch / bgrp;
	}
	return cost;
}

TCut::SchCost TCut::search(lid_t depth){
	//printf("\tT\t%d\t%d\t%d\t%d\n",depth,from,to,cluster.num_cores());
	SchCost tot_cost(0,0);
	len_t tot_bgrps = tot_batch / bgrp;
	if(depth <= 1){
		for(lid_t i=from;i<to;++i){
			sn_ptr p= new LNode(i, bgrp, cluster);
			children.push_back(p);
			SchCost c = p->search(0);
			// Add
			// TODO: optimize time.
			tot_cost += c;
		}
		return tot_cost;
	}
	--depth;
	// Search each in DP:
	// DP[l] = DP[l-k]+search(l-k+1,k)
	sn_vec* vecs = new sn_vec[to-from];
	SchCost* costs = new SchCost[to-from];
	sn_ptr p = new LNode(from, bgrp, cluster);
	vecs[0].push_back(p);
	costs[0] = p->search(depth);
	for (lid_t i=from+2;i<=to;++i) {
		//if(depth > 1) //printf("HERE\t%d\n", i);
		sn_vec& cur_vec = vecs[i-from-1];
		SchCost& cur_cost = costs[i-from-1];
		// Cur seg:[from,i)
		if(i<to){
			p = new SCut(from, i, bgrp, cluster);
			cur_vec.push_back(p);
			cur_cost = p->search(depth);
		}
		for (lid_t j=from+1;j<i;++j) {
			if(costs[j-from-1].cost(tot_bgrps) >= cost_inf) continue;
			if(j == i-1){
				p = new LNode(j,bgrp,cluster);
			}else{
				p = new SCut(j,i,bgrp,cluster);
			}

			// Use tot_cost as buffer
			tot_cost = p->search(depth);
			// Add
			// TODO: optimize time.
			tot_cost += costs[j-from-1];
			if(tot_cost.cost(tot_bgrps) < cur_cost.cost(tot_bgrps)){
				cur_cost = tot_cost;
				cur_vec = vecs[j-from-1];
				cur_vec.push_back(p);
			}
		}
	}
	tot_cost = costs[to-from-1];
	if(tot_cost.cost(tot_bgrps) < cost_inf){
		children = vecs[to-from-1];
	}
	delete[] vecs;
	delete[] costs;
	return tot_cost;
}
*/
/*
// TODO: remove static.
static double max_reltime;

cidx_t part_cores(access_t cur_ops, access_t rem_ops, cidx_t rem_cores){
	double x = cur_ops;
	x *= rem_cores;
	x /= cur_ops + rem_ops;
	//if(x<=1) return 1;
	//if(x>=rem_cores - 1) return rem_cores;
	cidx_t xm = static_cast<cidx_t>(floor(x));
	if(rem_cores * (x-xm) > x){
		++xm;
		if(rem_ops>max_reltime*(rem_cores - xm)) return 0;
	}else if(cur_ops>max_reltime*xm){
		return 0;
	}
	return xm;
}


void SCut::search(lid_t depth, len_t c_bgrp, SchCost& last_cost){
	// TODO: remove static.
	static access_t op_cnum[MAX_CHIPS];
	SchCost tot_cost(0,0);
	len_t tot_bgrps = tot_batch / bgrp;
	len_t nbgrp = bgrp/c_bgrp;
	cycle_t max_time=0;
	access_t total_ops = network->total_op(from, to);
	cidx_t num_cores = cluster.num_cores();
	if(depth <= 1){
		sn_vec cur_clist;
		// Naive method
		if(num_cores < to-from) return;
		for(lid_t i=from;i<to;++i){
			op_cnum[i] = network->total_op(i, i+1);
		}
		if(!cluster.try_alloc(op_cnum+from, static_cast<cidx_t>(to-from), total_ops)) return;
		for(lid_t i=from;i<to;++i){
			sn_ptr p= new LNode(i, c_bgrp,
				cluster.sub_cluster(static_cast<cidx_t>(i-from)));
			cur_clist.push_back(p);
			SchCost c = p->search(0);
			max_time = MAX(max_time, c.time);
			// Add
			// TODO: optimize time.
			tot_cost += c;
		}
		tot_cost.time += max_time * (nbgrp - 1);
		if(tot_cost.cost(tot_bgrps) < last_cost.cost(tot_bgrps)){
			children = cur_clist;
			last_cost = tot_cost;
		}
		return;
	}
	if(num_cores < 2) return;
	--depth;
	max_reltime = total_ops;
	max_reltime /= num_cores;
	max_reltime *= 1.0/Cluster::min_util;
	// Search each in DP:
	// DP[l] = DP[l-k]+search(l-k+1,k)
	sn_vec* vecs = new sn_vec[to-from];
	SchCost* costs = new SchCost[to-from];
	for(int i=0;i<to-from;++i){
		//printf("\t\t\t\t\t%f\t%d\n",0.01,costs[i].cost(tot_bgrps) >= cost_inf);
	}
	cycle_t* max_times = new cycle_t[to-from];
	cidx_t* remain_cores = new cidx_t[to-from];
	access_t cur_ops = network->total_op(from, from+1);
	access_t remain_ops = total_ops - cur_ops;
	cidx_t cores = part_cores(cur_ops,remain_ops,num_cores);
	sn_ptr p;
	if(cores > 0){
		remain_cores[0] = num_cores - cores;
		p = new LNode(from, c_bgrp, cluster.sub_cluster(0, cores));
		vecs[0].push_back(p);
		costs[0] = p->search(depth);
		max_times[0] = costs[0].time;
		costs[0].time *= nbgrp;
	}
	for (lid_t i=from+2;i<=to;++i) {
		sn_vec& cur_vec = vecs[i-from-1];
		SchCost& cur_cost = costs[i-from-1];
		cycle_t& cur_mtime = max_times[i-from-1];
		cidx_t& rem_cores = remain_cores[i-from-1];

		remain_ops = network->total_op(i, to);
		// Cur seg:[from,i)
		if(i<to){
			cur_ops = network->total_op(from, i);
			cores = part_cores(cur_ops,remain_ops,num_cores);
			if(cores > 0){
				p = new TCut(from, i, c_bgrp, cluster.sub_cluster(0, cores));
				cur_vec.push_back(p);
				cur_cost = p->search(depth);
				cur_mtime = cur_cost.time;
				cur_cost.time *= nbgrp;
				rem_cores = num_cores - cores;
			}
		}
		for (lid_t j=from+1;j<i;++j) {
			if(costs[j-from-1].cost(tot_bgrps) >= cost_inf) continue;
			cidx_t last_cores = remain_cores[j-from-1];
			if(i<to){
				if(last_cores <= 1) continue;
				cur_ops = network->total_op(j, i);
				cores = part_cores(cur_ops,remain_ops,last_cores);
				if(cores <= 0) continue;
			}else{
				cores = last_cores;
				//printf("C:%d,L:%d\n",cores,last_cores);
			}
			if(j == i-1){
				p = new LNode(j,c_bgrp, cluster.sub_cluster(num_cores - last_cores, cores));
			}else{
				p = new TCut(j,i,c_bgrp, cluster.sub_cluster(num_cores - last_cores, cores));
			}

			// Use tot_cost as buffer
			tot_cost = p->search(depth);
			// Add
			// TODO: optimize time.
			max_time = MAX(tot_cost.time, cur_mtime);
			tot_cost += costs[j-from-1];
			tot_cost.time += (nbgrp-1)*(max_time-cur_mtime);
			if(tot_cost.cost(tot_bgrps) < cur_cost.cost(tot_bgrps)){
				cur_cost = tot_cost;
				cur_mtime = max_time;
				cur_vec = vecs[j-from-1];
				cur_vec.push_back(p);
				rem_cores = last_cores - cores;
				//printf("SET:%d\n",rem_cores);
				//printf("Found\t%d\t%d\t%d\t%d\n",from, to,j,i);
			}
		}
	}
	tot_cost = costs[to-from-1];
	if(tot_cost.cost(tot_bgrps) < last_cost.cost(tot_bgrps)){
		children = vecs[to-from-1];
		last_cost = tot_cost;
	}
	delete[] vecs;
	delete[] costs;
	delete[] max_times;
	delete[] remain_cores;
	return;
}

SCut::SchCost SCut::search(lid_t depth){
	//printf("\tS\t%d\t%d\t%d\t%d\n",depth,from,to,cluster.num_cores());
	// For bgrp from small to large
	SchCost tot_cost;
	for (len_t c_bgrp = 1;c_bgrp <= bgrp;++c_bgrp) {
		if(bgrp % c_bgrp != 0) continue;
		search(depth, c_bgrp, tot_cost);
	}
	return tot_cost;
}
*/

SchNode::SchCost::SchCost(energy_t _energy, cycle_t _time)
	:energy(_energy),time(_time){}

cost_t SchNode::SchCost::cost(len_t nbatch) const{
	return calc_cost(energy, time*nbatch);
}

SchNode::SchCost&SchNode::SchCost::operator+=(const SchNode::SchCost& other){
	if(!(isValid() && other.isValid())){
		energy = energy_inf;
		return *this;
	}
	energy += other.energy;
	time += other.time;
	return *this;
}

SchNode::SchCost& SchNode::SchCost::operator*=(len_t other){
	if(isValid()){
		energy *= other;
		time *= other;
	}
	return *this;
}

bool SchNode::SchCost::operator!=(const SchNode::SchCost& other) const{
	if(isValid() && other.isValid())
		return energy != other.energy || time != other.time;
	return isValid() == other.isValid();
}

bool SchNode::SchCost::isValid() const{
	return energy < energy_inf;
}

std::ostream& operator<<(std::ostream& os, const SchNode::SchCost& cost){
	return os << "E:" << cost.energy << ", T:" << cost.time << ", Cost:" << cost.cost();
}

Json::Value SchNode::IR_gen() const{
	std::vector<Json::Value> workload_list;
	cidx_t num_cores = cluster.ylen * (cluster.xlen+2);
	workload_list.resize(num_cores);
	workload_cnt = 0;
	transferid_cnt = 0;
	root = this;
	wlid.resize(num_cores);
	for(size_t i=0;i<wlid.size();++i){
		wlid[i].resize(network->len());
		for(size_t j=0;j<wlid[i].size();++j){
			wlid[i][j].resize(tot_batch);
		}
	}
	ofmapid.resize(0);
	from_core.resize(0);
	weight_from_core.resize(0);
	to_dram.resize(0);
	curr_ifmap.resize(num_cores);
	curr_weight.resize(num_cores);
	name_to_id.clear();
	DRAM.clear();
	DRAM_ifmap_pos.clear();
	DRAM_weight_pos.clear();
	DRAM_ifmap_pos.clear();
	for(int i=0; i<network->len(); ++i){
		name_to_id[network->getNode(i).name()] = i;
	}
	add_workload_and_dfs(0, 0, workload_list);

	Json::Value ret;
	for(cidx_t i=0;i<num_cores;++i){
		for(const auto& ifmap:curr_ifmap[i]){
			Json::StyledWriter swriter;
			std::string IR_str = swriter.write(ifmap);
			std::cout << IR_str;
		}
		for(const auto& weight:curr_weight[i]){
			Json::StyledWriter swriter;
			std::string IR_str = swriter.write(weight);
			std::cout << IR_str;
		}
		assert(curr_ifmap[i].empty());
		assert(curr_weight[i].empty());

		Json::Value* last_wl = nullptr;
		for(Json::Value& wl: workload_list[i]){
			for(Json::Value& buffer: wl["buffer"]){
				if(buffer["type"] == "ifmap"){
					buffer["workload_id"] = workload_list[i][wlid[i][name_to_id[buffer["layer"].asString()]][buffer["lower"][0u].asUInt()]]["workload_id"];
					buffer["source"] = workload_list[i][wlid[i][name_to_id[buffer["layer"].asString()]][buffer["lower"][(Json::Value::UInt) 0].asUInt()]]["ifmap_temp"][buffer["layer"].asString()+"_"+std::to_string(buffer["lower"][(Json::Value::UInt) 0].asUInt())]["source"];
					for(Json::Value &source: buffer["source"]){
						buffer["transfer_id"].append(source["transfer_id"]);
					}
				}
				if(buffer["type"] == "weight"){
					if(buffer.isMember("from_core")){
						if(!buffer.isMember("workload_id")){
							buffer["workload_id"] = workload_list[i][wlid[i][name_to_id[buffer["layer"].asString()]][buffer["lower"][0u].asUInt()]]["workload_id"];
						}
						buffer.removeMember("from_core");
					}
					if(!buffer.isMember("source")){
						buffer["source"] = workload_list[i][wlid[i][name_to_id[buffer["layer"].asString()]][buffer["lower"][(Json::Value::UInt) 0].asUInt()]]["weight_temp"][buffer["layer"].asString()+"_"+std::to_string(buffer["lower"][(Json::Value::UInt) 0].asUInt())]["source"];
					}
					if(!buffer.isMember("transfer_id")){
						for(Json::Value &source: buffer["source"]){
							buffer["transfer_id"].append(source["transfer_id"]);
						}
					}
				}
			}
			if(wl.isMember("ifmap")){
				if(!from_core[wl["workload_id"].asUInt()]){
					Json::Value buffer;
					buffer["type"] = "ifmap";
					buffer["layer"] = wl["layer_name"];
					buffer["lower"] = wl["ifmap"]["lower"];
					buffer["upper"] = wl["ifmap"]["upper"];
					buffer["workload_id"] = wl["workload_id"];
					buffer["block"] = ((wl["ifmap"]["upper"][0u].asUInt() - wl["ifmap"]["lower"][0u].asUInt() + 1) * (wl["ifmap"]["upper"][1].asUInt() - wl["ifmap"]["lower"][1].asUInt() + 1) * (wl["ifmap"]["upper"][2].asUInt() - wl["ifmap"]["lower"][2].asUInt() + 1) * (wl["ifmap"]["upper"][3].asUInt() - wl["ifmap"]["lower"][3].asUInt() + 1) + 1023) >> 10;
					buffer["source"] = wl["ifmap_temp"][buffer["layer"].asString()+"_"+std::to_string(buffer["lower"][0u].asUInt())]["source"];
					for(Json::Value &source: buffer["source"]){
						buffer["transfer_id"].append(source["transfer_id"]);
					}
					//buffer["DRAMIFMAP"] = true;
					wl["buffer"].append(buffer);
					if(last_wl && (*last_wl)["workload_id"] >= wl["ifmap"]["max_workload_id"].asUInt()){
						(*last_wl)["buffer"].append(buffer);
					}
				}
			}
			if(wl.isMember("weight") && wl["weight"].isMember("from_ofmap")){
				if(!weight_from_core[wl["workload_id"].asUInt()]){
					Json::Value buffer;
					buffer["type"] = "weight";
					buffer["layer"] = wl["layer_name"];
					buffer["lower"] = wl["weight"]["lower"];
					buffer["upper"] = wl["weight"]["upper"];
					buffer["workload_id"] = wl["workload_id"];
					buffer["block"] = wl["weight"]["size"].asUInt() / 8 + 1023 >> 10;
					buffer["source"] = wl["weight_temp"][buffer["layer"].asString()+"_"+std::to_string(buffer["lower"][0u].asUInt())]["source"];
					for(Json::Value &source: buffer["source"]){
						buffer["transfer_id"].append(source["transfer_id"]);
					}
					wl["buffer"].append(buffer);
					if(last_wl && (*last_wl)["workload_id"] >= wl["weight"]["max_workload_id"].asUInt()){
						(*last_wl)["buffer"].append(buffer);
					}
				}
			}
			if(wl.isMember("ifmap_temp")){
				wl.removeMember("ifmap_temp");
			}
			if(wl.isMember("weight_temp")){
				wl.removeMember("weight_temp");
			}
			last_wl = &wl;
		}
		if(workload_list[i].type() != Json::nullValue){
			ret[std::to_string(i)]=workload_list[i];
		}
	}
	for(auto &in: DRAM["in"]){
		if(in.isMember("related_ofmap_map")){
			in.removeMember("related_ofmap_map");
		}
	}
	ret["top_batch_cut"] = root->type != SchNode::NodeType::L ? dynamic_cast<const Cut*>(root)->get_num_bgrp() : 1;
	ret["-1"] = DRAM;
	ret["xlen"] = cluster.xlen;
	ret["ylen"] = cluster.ylen;
	return ret;
}


void SCut::add_workload_and_dfs(len_t batch_offset, len_t segment, std::vector<Json::Value>& workload_list) const{
	const len_t stage_size = num_batch/num_bgrp;
	for(len_t stage_id=0; stage_id < num_bgrp+num_stage; ++stage_id){
		size_t i=0;
		for(auto child : children){
			if(stage_id >= stage[i] && stage_id < stage[i]+num_bgrp){
				len_t stage_offset = (stage_id - stage[i]) * stage_size + batch_offset;
				child->add_workload_and_dfs(stage_offset, segment, workload_list);
			}
			++i;
		}
	}
}

void TCut::add_workload_and_dfs(len_t batch_offset, len_t segment, std::vector<Json::Value>& workload_list) const{
	for(len_t i=0;i<num_batch;i+=num_batch/num_bgrp){
		for(auto& child : children){
			child->add_workload_and_dfs(batch_offset + i, segment, workload_list);
			if(this == root){
				segment++;
			}
		}
	}
}

const LNode* LNode::get_lnode_by_id(lid_t id) const{
	assert(contains(id));
	return this;
}

const LNode* Cut::get_lnode_by_id(lid_t id) const{
	for(auto child : children){
		if(child->contains(id))
			return child->get_lnode_by_id(id);
	}
	assert(0);
	return nullptr;
}

void LNode::add_workload_and_dfs(len_t batch_offset, len_t segment, std::vector<Json::Value>& workload_list) const{
	//printf("layer: %s, batch: %d\n", layert.name().c_str(), batch_offset);
	Json::Value empty_list;
	empty_list.append(1);
	empty_list.resize(0);
	const auto& ofm_parts = place_sch.getOfmL();
	for(auto part: ofm_parts){
		fmap_range range = part.first;
		if(range.is_empty()) continue;
		range.b += batch_offset;
		pos_t core = part.second;
		Cluster::xyid_t core_id = Cluster::get_xyid(core);
		Json::Value workload;
		workload["workload_id"] = workload_cnt++;
		workload["layer_name"] = layert.name();
		if(REF_IS_INSTANCE(layert.layer(), FCLayer)){
			workload["layer_type"] = "fc";
		}
		else if(REF_IS_INSTANCE(layert.layer(), ConvLayer)){
			workload["layer_type"] = "conv2d";
		}
		else if(REF_IS_INSTANCE(layert.layer(), PoolingLayer)){
			workload["layer_type"] = "pool";
		}
		else if(REF_IS_INSTANCE(layert.layer(), EltwiseLayer)){
			workload["layer_type"] = "element_wise";
		}
		else if(REF_IS_INSTANCE(layert.layer(), PTPLayer)){
			workload["layer_type"] = "point_to_point";
		}
		Json::Value oblock_lower, oblock_upper;
		oblock_lower.append(range.b.from);
		oblock_lower.append(range.c.from);
		oblock_lower.append(range.h.from);
		oblock_lower.append(range.w.from);

		oblock_upper.append(range.b.to - 1);
		oblock_upper.append(range.c.to - 1);
		oblock_upper.append(range.h.to - 1);
		oblock_upper.append(range.w.to - 1);

		workload["workload"].append(oblock_lower);
		workload["workload"].append(oblock_upper);
		workload["ofmap_size"] = range.size() * 8;

		workload["time"] = (int)tileSch.cost.time;

		if(REF_IS_INSTANCE(layert.layer(), ConvLayer) && !layert.hasWgtPrevs()){
			Json::Value weight;
			weight["lower"] = range.c.from;
			weight["upper"] = range.c.to - 1;
			Json::Value key;
			key["segment"] = segment;
			key["layer_name"] = layert.name();
			key["lower"] = weight["lower"];
			key["upper"] = weight["upper"];
			Json::Value destination;
			destination["type"] = "core";
			destination["id"] = core_id;
			destination["workload_id"] = workload["workload_id"];
			tfid_t transfer_id = 0;
			if(DRAM_weight_pos.count(key)){
				transfer_id = DRAM["out"][DRAM_weight_pos[key]]["transfer_id"].asUInt();
				len_t batch_size = 0;
				if(root->get_type() != NodeType::L){
					batch_size = tot_batch/dynamic_cast<const Cut*>(root)->get_num_bgrp();
				}
				else{
					batch_size = tot_batch;
				}
				if(batch_offset % batch_size == 0){
					DRAM["out"][DRAM_weight_pos[key]]["destination"].append(destination);
				}
			}
			else{
				DRAM_weight_pos[key] = DRAM["out"].size();
				transfer_id = transferid_cnt++;
				Json::Value dram_weight;
				dram_weight["destination"].append(destination);
				dram_weight["layer_name"] = layert.name();
				dram_weight["lower"] = weight["lower"];
				dram_weight["upper"] = weight["upper"];
				dram_weight["related_ifmap"] = empty_list;
				dram_weight["transfer_id"] = transfer_id;
				ConvLayer::Workload wl = static_cast<const ConvLayer&>(layert.layer()).get_workload();
				dram_weight["size"] = wl.R * wl.S * wl.C * range.c.size() * 8;
				dram_weight["type"] = "weight";
				DRAM["out"].append(dram_weight);
			}
			weight["transfer_id"].append(transfer_id);
			workload["weight"] = weight;
		}

		fmap_range ofmap_range = range;
		fmap_range weight_range = range;
		layert.layer().ofm_to_ifm(ofmap_range);
		layert.layer().ofm_to_wgt(weight_range);

		if(REF_IS_INSTANCE(layert.layer(), ConvLayer) && layert.hasWgtPrevs()){
			Json::Value weight;
			weight["lower"].append(weight_range.b.from);
			weight["lower"].append(weight_range.c.from);
			weight["lower"].append(weight_range.h.from);
			weight["upper"].append(weight_range.b.to-1);
			weight["upper"].append(weight_range.c.to-1);
			weight["upper"].append(weight_range.h.to-1);
			weight["from_ofmap"] = true;
			weight["size"] = weight_range.size() * 8;
			workload["weight"] = weight;
		}

		workload["ifmap"]["lower"].append(ofmap_range.b.from);
		workload["ifmap"]["lower"].append(ofmap_range.c.from);
		workload["ifmap"]["lower"].append(ofmap_range.h.from);
		workload["ifmap"]["lower"].append(ofmap_range.w.from);
		workload["ifmap"]["upper"].append(ofmap_range.b.to-1);
		workload["ifmap"]["upper"].append(ofmap_range.c.to-1);
		workload["ifmap"]["upper"].append(ofmap_range.h.to-1);
		workload["ifmap"]["upper"].append(ofmap_range.w.to-1);

		Bitset prev = layert.getPrevs();
		Bitset next = layert.get_nexts();
		wlid_t max_from_workload_id = 0;
		wlid_t weight_max_from_workload_id = 0;
		//std::cerr << prev;
		bool from_other_core = false, weight_from_other_core = false;
		len_t prev_channel_offset = 0;
		FOR_BITSET(layerno, prev){
			const Node& node = network->getNode(layerno);
			const LNode* lnode = root->get_lnode_by_id(layerno);
			assert(layert.getIfmPrevs().contains(layerno)^layert.getWgtPrevs().contains(layerno));
			const auto input_range = layert.getIfmPrevs().contains(layerno) ? ofmap_range : weight_range;
			const auto real_prev_channel_offset = layert.getIfmPrevs().contains(layerno) ? prev_channel_offset : 0;
			if(dirp_set.contains(layerno)){
				for(auto prev_part: lnode->get_place_sch().getOfmL()){
					for(len_t prev_batch_offset=0; prev_batch_offset<tot_batch; prev_batch_offset += lnode->num_batch){
						Cluster::xyid_t from_id = Cluster::get_xyid(prev_part.second);
						fmap_range prev_range = prev_part.first;
						prev_range.b += prev_batch_offset;
						prev_range.c += real_prev_channel_offset;
						/*if(layert.name() == "encoder1_QK" && !layert.getIfmPrevs().contains(layerno))
						{
							printf("prev_range_from = %d %d %d %d\n",prev_range.b.from,prev_range.c.from,prev_range.h.from,prev_range.w.from);
							printf("prev_range_to = %d %d %d %d\n",prev_range.b.to,prev_range.c.to,prev_range.h.to,prev_range.w.to);
							printf("input_range_from = %d %d %d %d\n",input_range.b.from,input_range.c.from,input_range.h.from,input_range.w.from);
							printf("input_range_to = %d %d %d %d\n",input_range.b.to,input_range.c.to,input_range.h.to,input_range.w.to);
							printf("batch_offset = %d\n\n",batch_offset);
						}*/
						fmap_range intersect = input_range.intersect(prev_range);
						if(intersect.is_empty())
							continue;
						prev_range.c -= real_prev_channel_offset;
						intersect.c -= real_prev_channel_offset;
						if(layert.getIfmPrevs().contains(layerno)){
							from_other_core = 1;
						}
						else{
							weight_from_other_core = 1;
						}
						Json::Value ifmap;
						ifmap["lower"].append(intersect.b.from);
						ifmap["lower"].append(intersect.c.from);
						ifmap["lower"].append(intersect.h.from);
						ifmap["lower"].append(intersect.w.from);

						ifmap["upper"].append(intersect.b.to-1);
						ifmap["upper"].append(intersect.c.to-1);
						ifmap["upper"].append(intersect.h.to-1);
						ifmap["upper"].append(intersect.w.to-1);

						ifmap["channel"].append(real_prev_channel_offset + intersect.c.from);
						ifmap["channel"].append(real_prev_channel_offset + intersect.c.to-1);

						/*vol_t ifmap_size = 1;
						for(int i=0; i<4; ++i){
							ifmap_size *= ifmap["source"]["upper"][i].asUInt() - ifmap["source"]["lower"][i].asUInt();
						}
						ifmap["source"]["size"] = ifmap_size * 8;*/

						ifmap["size"] = intersect.size()*8;

						ifmap["type"] = "core";
						ifmap["id"] = from_id;
						//ifmap["workload_id"] = workload_list[from_id][wlid[from_id][layerno][intersect.b.from]]["workload_id"];
						ifmap["layer_name"] = node.name();

						jsonindex_t prev_wlid = wlid[from_id][layerno][intersect.b.from];
						wlid_t prev_workload_id = workload_list[from_id][prev_wlid]["workload_id"].asUInt();
						if(layert.getIfmPrevs().contains(layerno)){
							max_from_workload_id = std::max(max_from_workload_id, prev_workload_id);
						}
						else{
							weight_max_from_workload_id = std::max(weight_max_from_workload_id, prev_workload_id);
						}
						if(ofmapid[prev_workload_id].count(intersect)){
							ifmap["transfer_id"] = workload_list[from_id][prev_wlid]["ofmap"][ofmapid[prev_workload_id][intersect]]["transfer_id"];
						}
						else{
							ifmap["transfer_id"] = transferid_cnt++;
						}

						if(layert.getIfmPrevs().contains(layerno)){
							workload["ifmap"]["transfer_id"].append(ifmap["transfer_id"]);
							workload["ifmap_temp"][layert.name()+"_"+std::to_string(range.b.from)]["source"].append(ifmap);
						}
						else{
							workload["weight"]["transfer_id"].append(ifmap["transfer_id"]);
							workload["weight_temp"][layert.name()+"_"+std::to_string(range.b.from)]["source"].append(ifmap);
						}
						

						Json::Value destination;
						destination["type"] = "core";
						destination["id"] = core_id;
						destination["workload_id"] = workload["workload_id"];
						destination["layer_name"] = layert.name();

						if(ofmapid[prev_workload_id].count(intersect)){
							workload_list[from_id][prev_wlid]["ofmap"][ofmapid[prev_workload_id][intersect]]["destination"].append(destination);
						}
						else{
							Json::Value ofmap;
							ofmap["lower"].append(intersect.b.from);
							ofmap["lower"].append(intersect.c.from);
							ofmap["lower"].append(intersect.h.from);
							ofmap["lower"].append(intersect.w.from);

							ofmap["upper"].append(intersect.b.to-1);
							ofmap["upper"].append(intersect.c.to-1);
							ofmap["upper"].append(intersect.h.to-1);
							ofmap["upper"].append(intersect.w.to-1);

							ofmap["transfer_id"] = ifmap["transfer_id"];
							ofmap["size"] = intersect.size()*8;

							ofmap["destination"].append(destination);
							ofmapid[prev_workload_id][intersect] = workload_list[from_id][prev_wlid]["ofmap"].size();
							workload_list[from_id][prev_wlid]["ofmap"].append(ofmap);
						}
					}
				}
			}
			else{
				int lower_c, upper_c;
				lower_c = std::max(0, (int)input_range.c.from - (int)real_prev_channel_offset);
				upper_c = std::min((int)node.layer().ofmap_shape().c, (int)input_range.c.to - (int)real_prev_channel_offset);
				if(lower_c < upper_c){
					Json::Value related_ifmap;
					Json::Value ifmap;
					ifmap["lower"].append(input_range.b.from);
					ifmap["lower"].append(lower_c);
					ifmap["lower"].append(input_range.h.from);
					ifmap["lower"].append(input_range.w.from);

					ifmap["upper"].append(input_range.b.to-1);
					ifmap["upper"].append(upper_c-1);
					ifmap["upper"].append(input_range.h.to-1);
					ifmap["upper"].append(input_range.w.to-1);

					ifmap["channel"].append(real_prev_channel_offset + lower_c);
					ifmap["channel"].append(real_prev_channel_offset + upper_c-1);

					vol_t ifmap_size = 1;
					for(int i=0; i<4; ++i){
						ifmap_size *= ifmap["upper"][i].asUInt() - ifmap["lower"][i].asUInt() + 1;
					}
					ifmap["size"] = ifmap_size * 8;

					ifmap["type"] = "DRAM";
					ifmap["id"] = 0;
					ifmap["layer_name"] = node.name();

					Json::Value key;
					key["lower"] = ifmap["lower"];
					key["upper"] = ifmap["upper"];
					key["source_layer_name"] = node.name();
					key["destination_layer_name"] = layert.name();
					key["type"] = layert.getIfmPrevs().contains(layerno) ? "ifmap" : "weight";

					tfid_t ofmap_transfer_id;

					if(DRAM_ofmap_pos.count(key)){
						ofmap_transfer_id = DRAM["out"][DRAM_ofmap_pos[key]]["transfer_id"].asUInt();
					}
					else{
						ofmap_transfer_id = transferid_cnt++;
					}

					for(auto prev_part: lnode->get_place_sch().getOfmL()){
						for(len_t prev_batch_offset=0; prev_batch_offset<tot_batch; prev_batch_offset += lnode->num_batch){
							Cluster::xyid_t from_id = Cluster::get_xyid(prev_part.second);
							fmap_range prev_range = prev_part.first;
							prev_range.b += prev_batch_offset;
							prev_range.c += real_prev_channel_offset;
							/*if(layert.name() == "encoder1_QK" && !layert.getIfmPrevs().contains(layerno))
							{
								printf("prev_range_from = %d %d %d %d\n",prev_range.b.from,prev_range.c.from,prev_range.h.from,prev_range.w.from);
								printf("prev_range_to = %d %d %d %d\n",prev_range.b.to,prev_range.c.to,prev_range.h.to,prev_range.w.to);
								printf("input_range_from = %d %d %d %d\n",input_range.b.from,input_range.c.from,input_range.h.from,input_range.w.from);
								printf("input_range_to = %d %d %d %d\n",input_range.b.to,input_range.c.to,input_range.h.to,input_range.w.to);
								printf("batch_offset = %d\n\n",batch_offset);
							}*/
							fmap_range intersect = input_range.intersect(prev_range);
							if(intersect.is_empty())
								continue;
							prev_range.c -= real_prev_channel_offset;
							intersect.c -= real_prev_channel_offset;
							tfid_t transfer_id;

							jsonindex_t prev_wlid = wlid[from_id][layerno][intersect.b.from];
							wlid_t prev_workload_id = workload_list[from_id][prev_wlid]["workload_id"].asUInt();
							if(layert.getIfmPrevs().contains(layerno)){
								max_from_workload_id = std::max(max_from_workload_id, prev_workload_id);
							}
							else{
								weight_max_from_workload_id = std::max(weight_max_from_workload_id, prev_workload_id);
							}
							//max_prev_wlid = std::max(max_prev_wlid, prev_workload_id);
							Json::Value destination;
							destination["type"] = "DRAM";
							destination["id"] = 0;
							destination["layer_name"] = layert.name();

							Json::Value source;
							source["lower"].append(prev_range.b.from);
							source["lower"].append(prev_range.c.from);
							source["lower"].append(prev_range.h.from);
							source["lower"].append(prev_range.w.from);

							source["upper"].append(prev_range.b.to-1);
							source["upper"].append(prev_range.c.to-1);
							source["upper"].append(prev_range.h.to-1);
							source["upper"].append(prev_range.w.to-1);
							source["core_id"] = from_id;
							source["workload_id"] = prev_workload_id;

							if(ofmapid[prev_workload_id].count(prev_range)){
								transfer_id = workload_list[from_id][prev_wlid]["ofmap"][ofmapid[prev_workload_id][prev_range]]["transfer_id"].asUInt();
								source["transfer_id"] = transfer_id;
								if(!SchNode::to_dram[prev_workload_id]){
									workload_list[from_id][prev_wlid]["ofmap"][ofmapid[prev_workload_id][prev_range]]["destination"].append(destination);
									SchNode::to_dram[prev_workload_id] = true;
									DRAM_ifmap_pos[transfer_id] = DRAM["in"].size();
									DRAM["in"].append(source);
								}

							}
							else{
								transfer_id = transferid_cnt++;
								source["transfer_id"] = transfer_id;
								Json::Value ofmap;
								ofmap["lower"] = source["lower"];
								ofmap["upper"] = source["upper"];
								ofmap["destination"].append(destination);
								ofmap["transfer_id"] = transfer_id;
								ofmap["size"] = prev_range.size() * 8;
								SchNode::to_dram[workload_list[from_id][prev_wlid]["workload_id"].asUInt()] = true;
								ofmapid[prev_workload_id][prev_range] = workload_list[from_id][prev_wlid]["ofmap"].size();
								workload_list[from_id][prev_wlid]["ofmap"].append(ofmap);
								DRAM_ifmap_pos[transfer_id] = DRAM["in"].size();
								DRAM["in"].append(source);
							}
							if(!DRAM["in"][DRAM_ifmap_pos[transfer_id]]["related_ofmap_map"].isMember(std::to_string(ofmap_transfer_id))){
								DRAM["in"][DRAM_ifmap_pos[transfer_id]]["related_ofmap"].append(ofmap_transfer_id);
								DRAM["in"][DRAM_ifmap_pos[transfer_id]]["related_ofmap_map"][std::to_string(ofmap_transfer_id)] = true;
							}
							related_ifmap.append(transfer_id);
						}
					}
					Json::Value destination;
					destination["id"] = core_id;
					destination["type"] = "core";
					destination["workload_id"] = workload["workload_id"];

					if(DRAM_ofmap_pos.count(key)){
						DRAM["out"][DRAM_ofmap_pos[key]]["destination"].append(destination);
					}
					else{
						DRAM_ofmap_pos[key] = DRAM["out"].size();
						Json::Value ofmap;
						ofmap["lower"] = key["lower"];
						ofmap["upper"] = key["upper"];
						ofmap["layer_name"] = layert.name();
						ofmap["size"] = ifmap["size"];
						ofmap["transfer_id"] = ofmap_transfer_id;
						ofmap["destination"].append(destination);
						ofmap["related_ifmap"] = related_ifmap;
						ofmap["type"] = "fmap";
						DRAM["out"].append(ofmap);
					}
					ifmap["transfer_id"] = ofmap_transfer_id;
					if(layert.getIfmPrevs().contains(layerno)){
						workload["ifmap"]["transfer_id"].append(ofmap_transfer_id);
						workload["ifmap_temp"][layert.name()+"_"+std::to_string(range.b.from)]["source"].append(ifmap);
					}
					else{
						workload["weight"]["transfer_id"].append(ofmap_transfer_id);
						workload["weight_temp"][layert.name()+"_"+std::to_string(range.b.from)]["source"].append(ifmap);
					}
					
				}
			}
			if(layert.getIfmPrevs().contains(layerno)){
				prev_channel_offset += network->getNode(layerno).layer().ofmap_shape().c;
			}
			//fprintf(stderr,"layerno = %d, add = %d\n", layerno, network->getNode(layerno).layer().ofmap_shape().c);
			if(REF_IS_INSTANCE(layert.layer(), EltwiseLayer)){
				auto eltlayer = dynamic_cast<const EltwiseLayer*>(&(layert.layer()));
				prev_channel_offset %= eltlayer->get_workload().K;

			}
		}

		workload["ifmap"]["max_workload_id"] = max_from_workload_id;
		if(layert.hasWgtPrevs()){
			workload["weight"]["max_workload_id"] = weight_max_from_workload_id;
		}

		if(prev.count() == 0){
			Json::Value ifmap;
			ifmap["lower"].append(ofmap_range.b.from);
			ifmap["lower"].append(ofmap_range.c.from);
			ifmap["lower"].append(ofmap_range.h.from);
			ifmap["lower"].append(ofmap_range.w.from);

			ifmap["upper"].append(ofmap_range.b.to-1);
			ifmap["upper"].append(ofmap_range.c.to-1);
			ifmap["upper"].append(ofmap_range.h.to-1);
			ifmap["upper"].append(ofmap_range.w.to-1);

			ifmap["channel"].append(ofmap_range.c.from);
			ifmap["channel"].append(ofmap_range.c.to-1);
			ifmap["size"] = ofmap_range.size() * 8;

			ifmap["layer_name"] = "input";
			ifmap["id"] = 0;
			ifmap["type"] = "DRAM";

			Json::Value key;
			key["source_layer_name"] = "input";
			key["destination_layer_name"] = layert.name();
			key["lower"] = ifmap["lower"];
			key["upper"] = ifmap["upper"];

			tfid_t transfer_id;
			if(DRAM_ofmap_pos.count(key)){
				transfer_id = DRAM["out"][DRAM_ofmap_pos[key]]["transfer_id"].asUInt();
			}
			else{
				transfer_id = transferid_cnt++;
			}
			ifmap["transfer_id"] = transfer_id;
			workload["ifmap_temp"][layert.name()+"_"+std::to_string(range.b.from)]["source"].append(ifmap);
			Json::Value destination;
			destination["id"] = core_id;
			destination["type"] = "core";
			destination["workload_id"] = workload["workload_id"];

			if(DRAM_ofmap_pos.count(key)){
				DRAM["out"][DRAM_ofmap_pos[key]]["destination"].append(destination);
			}
			else{
				DRAM_ofmap_pos[key] = DRAM["out"].size();
				Json::Value input;
				input["transfer_id"] = transfer_id;
				input["layer_name"] = layert.name();
				input["related_ifmap"] = empty_list;
				input["destination"].append(destination);
				input["size"] = ifmap["size"];
				input["lower"] = ifmap["lower"];
				input["upper"] = ifmap["upper"];
				input["type"] = "fmap";
				DRAM["out"].append(input);
			}
			workload["ifmap"]["transfer_id"].append(transfer_id);
		}
		if(next.count() == 0){
			Json::Value ofmap;
			ofmap["lower"].append(range.b.from);
			ofmap["lower"].append(range.c.from);
			ofmap["lower"].append(range.h.from);
			ofmap["lower"].append(range.w.from);

			ofmap["upper"].append(range.b.to-1);
			ofmap["upper"].append(range.c.to-1);
			ofmap["upper"].append(range.h.to-1);
			ofmap["upper"].append(range.w.to-1);

			ofmap["size"] = range.size() * 8;
			ofmap["transfer_id"] = transferid_cnt++;

			Json::Value destination;
			destination["type"] = "DRAM";
			destination["id"] = 0;
			destination["layer_name"] = "output";
			ofmap["destination"].append(destination);
			workload["ofmap"].append(ofmap);

			Json::Value output;
			output["lower"] = ofmap["lower"];
			output["upper"] = ofmap["upper"];
			output["core_id"] = core_id;
			output["related_ofmap"] = empty_list;
			output["workload_id"] = workload["workload_id"];
			output["transfer_id"] = ofmap["transfer_id"];

			DRAM["in"].append(output);
		}

		bool to_other_core = false;

		FOR_BITSET(layerno, next){
			const Node& node = network->getNode(layerno);
			const LNode* lnode = root->get_lnode_by_id(layerno);
			len_t next_channel_offset = 0;
			const Bitset& prev_set = node.getIfmPrevs();
			if(lnode->layert.getIfmPrevs().contains(layerid)){
				FOR_BITSET(lid, prev_set){
					if(lid != layerid){
						next_channel_offset += network->getNode(lid).layer().ofmap_shape().c;
					}
					else{
						break;
					}
				}
			}
			if(REF_IS_INSTANCE(node.layer(), EltwiseLayer)){
				auto eltlayer = dynamic_cast<const EltwiseLayer*>(&(node.layer()));
				next_channel_offset %= eltlayer->get_workload().K;
			}
			if(lnode->get_dirp_set().contains(layerid)){
				for(auto next_part: lnode->get_place_sch().getOfmL()){
					for(len_t next_batch_offset=0; next_batch_offset<tot_batch; next_batch_offset += lnode->num_batch){
						Cluster::xyid_t to_id = Cluster::get_xyid(next_part.second);
						fmap_range next_range = next_part.first;
						if(next_range.is_empty()) continue;
						next_range.b += next_batch_offset;
						if(lnode->getLayer().getIfmPrevs().contains(layerid)){
							node.layer().ofm_to_ifm(next_range);
						}
						else{
							node.layer().ofm_to_wgt(next_range);
						}
						range.c += next_channel_offset;
						fmap_range intersect = range.intersect(next_range);
						range.c -= next_channel_offset;
						if(intersect.is_empty())
							continue;
						if(to_id != core_id){
							to_other_core = true;
						}
						Json::Value buffer;
						buffer["type"] = lnode->getLayer().getIfmPrevs().contains(layerid) ? "ifmap" : "weight";
						if(!lnode->getLayer().getIfmPrevs().contains(layerid)) 
							buffer["from_core"] = true;
						buffer["layer"] = node.layer().get_name();
						buffer["lower"].append(next_range.b.from);
						buffer["lower"].append(next_range.c.from);
						buffer["lower"].append(next_range.h.from);
						buffer["lower"].append(next_range.w.from);
						buffer["upper"].append(next_range.b.to-1);
						buffer["upper"].append(next_range.c.to-1);
						buffer["upper"].append(next_range.h.to-1);
						buffer["upper"].append(next_range.w.to-1);
						buffer["block"] = ((next_range.size() + 1023) >> 10);
						curr_ifmap[to_id].insert(buffer);
					}
				}
			}
		}

		len_t batch_size = 0;
		Json::Value weight;
		if(REF_IS_INSTANCE(layert.layer(), ConvLayer) && !layert.hasWgtPrevs()){
			if(root->get_type() != NodeType::L){
				batch_size = tot_batch/dynamic_cast<const Cut*>(root)->get_num_bgrp();
			}
			else{
				batch_size = tot_batch;
			}
			weight["type"] = "weight";
			weight["layer"] = layert.name();
			weight["lower"] = range.c.from;
			weight["upper"] = range.c.to - 1;
			ConvLayer::Workload wl = static_cast<const ConvLayer&>(layert.layer()).get_workload();
			Json::Value source;
			source["size"] = wl.R * wl.S * wl.C * range.c.size() * 8;
			weight["block"] = (source["size"].asUInt() / 8 + 1023) >> 10;
			source["id"] = 0;
			source["type"] = "DRAM";
			source["lower"] = weight["lower"];
			source["upper"] = weight["upper"];
			source["transfer_id"] = workload["weight"]["transfer_id"][0u];
			weight["source"].append(source);
			weight["transfer_id"].append(workload["weight"]["transfer_id"]);
			if(batch_offset % batch_size == 0){
				curr_weight[core_id].insert(weight);
				if(workload_list[core_id].size() && get_lca(this, root->get_lnode_by_id(name_to_id[workload_list[core_id][workload_list[core_id].size()-1]["layer_name"].asString()])) != root){
					if(workload_list[core_id][workload_list[core_id].size()-1]["layer_name"] != layert.name()){
						workload_list[core_id][workload_list[core_id].size()-1]["buffer"].append(weight);
					}
				}
			}
		}

		for(const Json::Value& datablock : curr_ifmap[core_id]){
			workload["buffer"].append(datablock);
		}
		for(const Json::Value& weight : curr_weight[core_id]){
			workload["buffer"].append(weight);
		}
		if(to_other_core || to_dram){
			Json::Value ofmap;
			ofmap["type"] = "ofmap";
			ofmap["layer"] = layert.name();
			ofmap["lower"].append(range.b.from);
			ofmap["lower"].append(range.c.from);
			ofmap["lower"].append(range.h.from);
			ofmap["lower"].append(range.w.from);
			ofmap["upper"].append(range.b.to-1);
			ofmap["upper"].append(range.c.to-1);
			ofmap["upper"].append(range.h.to-1);
			ofmap["upper"].append(range.w.to-1);
			//ofmap["block"] = (range.size() + 10239) / 10240;
			ofmap["block"] = 10;
			ofmap["size"] = range.size() * 8;
			workload["buffer"].append(ofmap);
		}

		for(len_t batch=range.b.from; batch<range.b.to; ++batch)
			wlid[core_id][layerid][batch] = workload_list[core_id].size();

		workload_list[core_id].append(workload);

		std::vector<Json::Value> this_workload_ifmap;
		for(const Json::Value& ifmap: curr_ifmap[core_id]){
			if(ifmap["layer"] == layert.name() && ifmap["lower"][0u] == range.b.from){
				this_workload_ifmap.push_back(ifmap);
			}
		}
		for(const Json::Value& ifmap: this_workload_ifmap){
			curr_ifmap[core_id].erase(ifmap);
		}
		if(REF_IS_INSTANCE(layert.layer(), ConvLayer) && !layert.hasWgtPrevs()){
			if((batch_offset + num_batch) % batch_size == 0){
				for(auto weight : curr_weight[core_id]){
					if(weight["layer"] == layert.name()){
						curr_weight[core_id].erase(weight);
						break;
					}
				}
			}
		}
		ofmapid.resize(workload_cnt);
		from_core.push_back(from_other_core);
		weight_from_core.push_back(weight_from_other_core);
		SchNode::to_dram.push_back(false);
	}
}
