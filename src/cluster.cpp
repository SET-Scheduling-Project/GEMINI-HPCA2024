#include "cluster.h"
#include "noc.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

mlen_t Cluster::xlen,Cluster::ylen,Cluster::stride;
double Cluster::min_util;
//cidx_t Cluster::tmp_nlist[MAX_CHIPS];

Cluster::Cluster(cidx_t _first, cidx_t _last){
	use_range = true;
	range.first = _first;
	range.last = _last;
	//printf("%d\t%d\n",first, last);
	assert(range.last > range.first);
}
Cluster::Cluster(std::vector<cidx_t> &_tilelist) {
	use_range = false;
	core_list = _tilelist;
}

bool Cluster::operator==(const Cluster& other) const{
	if (use_range) {
		return range.first == other.range.first && range.last == other.range.last;
	}
	else {
		return core_list == other.core_list;
	}

}

bool Cluster::operator!=(const Cluster& other) const{
	return !operator==(other);
}

cidx_t Cluster::num_cores() const{
	if (use_range) {
		return range.last - range.first;
	}
	else {
		return core_list.size();
	}
}
void Cluster::change_to_list() {
	assert(use_range);
	cidx_t num = num_cores();
	core_list.reserve(num);
	for (cidx_t i = 0; i < num; i++) {
		core_list.push_back(range.first + i);
	}
	use_range = false;
}

void Cluster::set_core_set(){
	if (core_set.empty()&&use_range) {
		cidx_t num = num_cores();
		for (cidx_t i = 0; i < num; i++) {
			core_set.insert(range.first + i);
		}
	}
	assert(!core_set.empty()||!core_list.empty());
}

Cluster::allocRes_t Cluster::try_alloc(utime_t* ops, cidx_t childNum, utime_t totOps, bool base) const {
	cidx_t totalNodes = num_cores();
	if (childNum > totalNodes) return nullptr;
	if (totOps <= 0) {
		totOps = 0;
		for (cidx_t i = 0; i < childNum; ++i) {
			totOps += ops[i];
		}
	}

	// TODO: (low priority) check whether changing this to stack saves time.
	auto* ratioList = new std::pair<double, cidx_t>[childNum];
	auto* diff = new std::pair<double, cidx_t>[childNum];
	auto* isPlaced = new bool[childNum]();
	auto* nodes_raw = new utime_t[childNum];
	Cluster::allocRes_t allocRes = std::make_unique<cidx_t[]>(childNum + 1);
	double max_time = 0;
	// memset(isPlaced, 0, sizeof(bool)*static_cast<size_t>(childNum));
	// memset(tmp_nlist, 0, sizeof(cidx_t)*static_cast<size_t>(childNum));
	if (!base) {
		cidx_t remainNodes = totalNodes;
		utime_t remainOps = totOps;


		while (remainOps > 0) {
			assert(remainNodes > 0);
			/*
			if (remainNodes == 0){
				delete[] isPlaced;
				delete[] ratioList;
				delete[] allocRes;
				return nullptr;
			}
			*/
			cidx_t j = 0;
			cidx_t addNodes = remainNodes;
			for (cidx_t i = 0; i < childNum; ++i) {
				if (isPlaced[i]) continue;
				double idealNodes = (static_cast<double>(ops[i]) / remainOps) * remainNodes;
				allocRes[i] = static_cast<cidx_t>(idealNodes); // Floor by default.
				addNodes -= allocRes[i];
				ratioList[j++] = std::make_pair(allocRes[i] / idealNodes, i);
			}
			if (addNodes == 0) {
				for (cidx_t i = 0; i < childNum; ++i) {
					if (isPlaced[i]) continue;
					assert(allocRes[i] > 0);
					double cur_mtime = ops[i] / static_cast<double>(allocRes[i]);
					max_time = MAX(max_time, cur_mtime);
				}
				break;
			}
			assert(addNodes > 0);
			std::sort(ratioList, ratioList + j);
			j = 0;
			do {
				cidx_t cur_idx = ratioList[j++].second;
				remainNodes -= ++allocRes[cur_idx];
				remainOps -= ops[cur_idx];
				isPlaced[cur_idx] = true;
				double cur_mtime = ops[cur_idx] / static_cast<double>(allocRes[cur_idx]);
				max_time = MAX(max_time, cur_mtime);
			} while (--addNodes > 0);
		}
		/*
		double utilization = totOps / (totalNodes * max_time);
		assert(utilization < 1 + 1e-6);
		if (utilization < min_util) {
			allocRes.reset();
			return allocRes;
		}*/
	}
	else {
		cidx_t totalNodes_ = 0;

		for (cidx_t i = 0; i < childNum; ++i) {
			nodes_raw[i] = ops[i] / totOps * static_cast<double>(totalNodes);
			allocRes[i] = MAX(1, floor(nodes_raw[i] / static_cast<double>(stride))) * stride;
			totalNodes_ += allocRes[i];
			//			std::cout << "totalNodes_ = " << totalNodes_;
		}
		do {
			for (cidx_t i = 0; i < childNum; ++i) {
				diff[i].first = double(allocRes[i]) - nodes_raw[i];
				diff[i].second = i;
			}
			std::sort(diff, diff + childNum - 1);
			if (totalNodes_ > totalNodes) {
				allocRes[diff[childNum - 1].second] -= stride;
				totalNodes_ -= stride;
			}
			else if (totalNodes_ < totalNodes) {
				allocRes[diff[0].second] += stride;
				totalNodes_ += stride;
			}
		} while (totalNodes_ != totalNodes);
	}
	delete[] isPlaced;
	delete[] ratioList;
	delete[] nodes_raw;
	delete[] diff;
	for (cidx_t i = 0; i < childNum; ++i) {
		if (allocRes[i] <= 0) {
			//std::cout << "allocRes[i] <= 0, i = " << i << " value = " << allocRes[i] << "\n";
			allocRes.reset();
			return allocRes;
		}
	}
	cidx_t curTNum = allocRes[0], nextTNum;

	allocRes[0] = range.first;
	double maxtime = 0;
	for (cidx_t i = 0; i < childNum; ++i) {
		nextTNum = allocRes[i + 1];
		allocRes[i + 1] = allocRes[i] + curTNum;
		maxtime = (ops[i] / static_cast<double>(curTNum))< maxtime ? maxtime: (ops[i] / static_cast<double>(curTNum));
		curTNum = nextTNum;		//std::cout<<"allocres i = "<<i<< "id" << allocRes[i]<<"\n";
	}
	/*
	double utilization = totOps / (totalNodes * maxtime);
	//if (utilization > 1 + 1e-6) {
	//	assert(true);
	//}
	assert(utilization < 1 + 1e-6);
	if (utilization < min_util) {
		allocRes.reset();
		return allocRes;
	}*/
	assert(allocRes[childNum] == range.last);
	return allocRes;
}
Cluster Cluster::sub_cluster(std::vector<cidx_t>& _id_list) const {
	std::vector<cidx_t> temp_id;
	temp_id.reserve(num_cores());
	for (auto p : core_list)
		temp_id.push_back(p);
	return Cluster(temp_id);
}

Cluster Cluster::sub_cluster(cidx_t childIdx, const allocRes_t& allocRes) const{
	return Cluster(allocRes[childIdx], allocRes[childIdx+1]);
}

Cluster Cluster::sub_cluster(cidx_t from, cidx_t num) const{
	from += range.first;
	assert(from + num <= range.last);
	return Cluster(from, from+num);
}
void Cluster::erase(std::vector<cidx_t>& _erase_list) {
	for (auto p : _erase_list)
		core_list.erase(core_list.begin() + p);
}
void Cluster::insert(std::vector<cidx_t>& _insert_loc, std::vector<cidx_t>& _insert_value) {
	for (cidx_t i = 0; i < _insert_loc.size(); i++)
		core_list.insert(core_list.begin () + _insert_loc[i], _insert_value[i]);
}
void Cluster::erase(cidx_t _erase_element) {
		core_list.erase(core_list.begin() + _erase_element);
}
void Cluster::insert(cidx_t _insert_loc, cidx_t _insert_value) {
		core_list.insert(core_list.begin() + _insert_loc, _insert_value);
}

pos_t Cluster::operator[](cidx_t num_chip) const{
	assert(num_chip < num_cores());
	if(use_range){
		return get_pos(range.first + num_chip);
	}
	return get_pos(core_list[num_chip]);
}

memidx_t Cluster::nearest_dram(){
	if (use_range) {
		set_core_set();
	}
	cidx_t x_sum = 0;
	cidx_t y_sum = 0;
	double center_x = 0;
	double center_y = 0;
	mlen_t ddr_id=0;
	auto num = core_set.size();
	if (use_range) {
		for (auto core_id : core_set) {
			x_sum += get_pos(core_id).x;
			y_sum += get_pos(core_id).y;
		}
	}
	else {
		for (auto core_id : core_list) {
			x_sum += get_pos(core_id).x;
			y_sum += get_pos(core_id).y;
		}
	}
	center_x = static_cast<double>(x_sum) / num;
	center_y = static_cast<double>(y_sum) / num;
	double min_dis = __DBL_MAX__;
	for (auto i = 0; i < NoC::DRAM_num; ++i) {
		double dis = 0;
		for (auto j : NoC::dram_list[i]) {
			dis += (abs(center_x - static_cast<double>(j.x)) + abs(center_y - static_cast<double>(j.y)));
		}
		if (dis < min_dis) {
			ddr_id = i;
			min_dis = dis;
		}
	}
	return ddr_id;
}

/*
Cluster::hop_t Cluster::unicast(pos_t src, pos_t dst){
	return static_cast<hop_t>(abs(src.x-dst.x)+abs(src.y-dst.y));
}

static inline Cluster::hop_t calc_intd(mlen_t s, mlen_t c, mlen_t b){
	return static_cast<Cluster::hop_t>(MAX(c,b) - MIN(s,c));
}

// TODO: better realization.
Cluster::hop_t Cluster::multicast(pos_t src, pos_t* dst, cidx_t len){
	mlen_t cur_x = dst[0].x;
	mlen_t min_y = dst[0].y;
	hop_t h = calc_intd(dst[0].x, src.x, dst[len-1].x);
	for(cidx_t i=1; i<len; ++i){
		if(dst[i].x != cur_x){
			cur_x = dst[i].x;
			h+=calc_intd(min_y, src.y, dst[i-1].y);
			min_y = dst[i].y;
		}
	}
	h+=calc_intd(min_y, src.y, dst[len-1].y);
	return h;
}

Cluster::hop_t Cluster::unicast_dram(pos_t dst, vol_t size){
	return (static_cast<hop_t>((xlen+1)*ylen+(ylen-1)*(ylen-2*dst.y)+2*dst.y*dst.y)*size)/static_cast<hop_t>(2*ylen);
}

Cluster::hop_t Cluster::unicast_to_dram(pos_t dst, vol_t size){
	return unicast_dram(dst, size);
}

// TODO: better multicast.
// TODO: multicast can also do unicast.
Cluster::hop_t Cluster::multicast_dram(pos_t* dst, cidx_t len, vol_t size){
	hop_t tot_hop=0;
	for(cidx_t i=0; i<len; ++i){
		tot_hop += unicast_dram(dst[i], size);
	}
	return tot_hop;
}
*/
pos_t Cluster::get_pos(cidx_t core_idx){
	assert(core_idx >= 0 && core_idx <xlen*ylen);
	cidx_t str_id = (core_idx / (stride*ylen));
	bool up_down = str_id%2==1;
	mlen_t x = str_id*stride+core_idx%stride;
	mlen_t y = (core_idx / stride) % ylen;
	y = up_down?((ylen-1)-y):y;
	return {x, y};
}

Cluster::xyid_t Cluster::get_xyid(pos_t& chip){
	return chip.y*(xlen+2)+chip.x+1;
}
