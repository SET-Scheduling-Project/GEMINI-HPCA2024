#include "noc.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include "schnode.h"
#include "util.h"

energy_t NoC::NoC_hop_cost;
energy_t NoC::NoP_hop_cost;
energy_t NoC::DRAM_acc_cost;
bw_t NoC::DRAM_bw;
bw_t NoC::NoC_bw;
bw_t NoC::NoP_bw;
bool NoC::interleave;
bool NoC::seperate_IO;
bool NoC::soc;
bool NoC::calc_noc_control;
mlen_t NoC::DRAM_num;
mlen_t NoC::DRAM_router_num;
mlen_t NoC::x_cut;
mlen_t NoC::y_cut;
mlen_t NoC::x_step;
mlen_t NoC::y_step;

std::vector<std::vector<pos_t>> NoC::dram_list;
std::vector<pos_t> NoC::dram_list_base;
std::unordered_set<noc_t> NoC::NoP_links;



NoC::NoC(bool _calc_bw,bool base): calc_bw(_calc_bw),is_base(base), NoC_tot_hops(0),NoP_tot_hops(0),core_hops(0), tot_DRAM_acc(0){
	DRAM_access.resize(NoC::DRAM_num);
	if (!is_base) {
		for (auto i = 0; i < DRAM_num; ++i) {
			DRAM_access[i] = 0;
		}
	}
}

	 

/*
void NoC::set_calc_bw(bool _calc_bw){
	calc_bw = _calc_bw;
	if(!calc_bw) link_hops.clear();
}
*/

void NoC::reset(){
	NoC_tot_hops = 0;
	NoP_tot_hops = 0;
	core_hops = 0;
	tot_DRAM_acc = 0;
	DRAM_access.resize(NoC::DRAM_num);
	if (!is_base) {
		for (auto i = 0; i < DRAM_num; ++i) {
			DRAM_access[i] = 0;
		}
	}
	link_hops.clear();
}

void NoC::fromRemoteMem(const DataLayout& toLayout, mlen_t dram_id){
	auto rLen = toLayout.rangeLength();
	for(cidx_t i=0; i<rLen; ++i){
		auto it = toLayout.at(i);
		vol_t curSize = it.range.size();
		if(curSize <= 0) continue;
		if(it.numTile == 1){
			unicast_dram(it.tiles[0], curSize,dram_id);
		}else{
			multicast_dram(it.tiles, it.numTile, curSize,dram_id);
		}
	}
}

void NoC::fromRemoteMem(const DataLayout& toLayout, len_t fromC, len_t toC, mlen_t dram_id){
	if(toC <= fromC) return;
	fmap_range::dim_range truncRange = {fromC, toC};

	auto rLen = toLayout.rangeLength();
	for(cidx_t i=0; i<rLen; ++i){
		auto it = toLayout.at(i);
		fmap_range range = it.range;
		range.c = range.c.intersect(truncRange);
		vol_t curSize = range.size();
		if(curSize <= 0) continue;
		if(it.numTile == 1){
			unicast_dram(it.tiles[0], curSize, dram_id);
		}else{
			multicast_dram(it.tiles, it.numTile, curSize,dram_id);
		}
	}
}

void NoC::toRemoteMem(const UniqueLayout& fromLayout, mlen_t dram_id){
	for(cidx_t i=0; i<fromLayout.totLength(); ++i){
		auto it = fromLayout[i];
		vol_t curSize = it.range.size();
		if(curSize <= 0) continue;
		unicast_to_dram(it.tile, curSize,dram_id);
	}
}

void NoC::betweenLayout(const UniqueLayout& fromLayout, const DataLayout& toLayout, len_t fromCOffset, len_t fromB, len_t toB, bool unordered){
	hop_t h = 0;
	// TODO: change to generic UniqueLayout
	const auto* fLayout = dynamic_cast<const StdULayout*>(&fromLayout);
	if(fLayout == nullptr){
		// TODO: add general case.
		assert(false);
		return;
	}
	bool diffB = (fromB != toB);
	auto rLen = toLayout.rangeLength();
	for(cidx_t i=0; i<rLen; ++i){
		auto toEntry = toLayout.at(i);
		fmap_range toRange = toEntry.range;
		if(toRange.c.to <= fromCOffset) continue;
		toRange.c -= fromCOffset;
		if(unordered){
			for(auto it :fLayout->get_intersect_bruteforce(toRange, fromB, toB)){
				auto fromEntry = it;
				vol_t v = calc_intersect(fromEntry.first, toRange, fromB, toB);
				if(v == 0) continue;
				if(toEntry.numTile == 1){
					h += unicastCalc(fromEntry.second, *toEntry.tiles, v);
				}else{
					h += multicastCalc(fromEntry.second, toEntry.tiles, toEntry.numTile, v);
				}
			}
		}
		else{
			for(auto it = fLayout->get_intersect(toRange, diffB);it.isValid();it.next()){
				auto fromEntry = *it;
				vol_t v = calc_intersect(fromEntry.range, toRange, fromB, toB);
				if(v == 0) continue;
				if(toEntry.numTile == 1){
					h += unicastCalc(fromEntry.tile, *toEntry.tiles, v);
				}else{
					h += multicastCalc(fromEntry.tile, toEntry.tiles, toEntry.numTile, v);
				}
			}
		}
	}
	if(fromB > toB)  h /= (fromB / toB);
	NoC_tot_hops += h;
	core_hops += h;
}

NoC NoC::operator+(const NoC& other) const{
	NoC x = *this;
	return x += other;
}

NoC& NoC::operator+=(const NoC& other){
	
	NoC_tot_hops += other.NoC_tot_hops;
	NoP_tot_hops += other.NoP_tot_hops;
	core_hops += other.core_hops;
	tot_DRAM_acc += other.tot_DRAM_acc;
	if (!is_base) {
		for (auto i = 0; i < DRAM_num; ++i) {
			DRAM_access[i] += other.DRAM_access[i];
		}
	}
	if(calc_bw || other.calc_bw){
		assert(calc_bw && other.calc_bw);
		link_hops += other.link_hops;
	}
	return *this;
}

NoC NoC::operator*(const len_t& batch) const{
	NoC x = *this;
	return x *= batch;
}

NoC& NoC::operator*=(const len_t& batch){
	NoC_tot_hops *= batch;
	NoP_tot_hops *= batch;
	core_hops *= batch;
	tot_DRAM_acc *= batch;
	if (!is_base) {
		for (auto i = 0; i < DRAM_num; ++i) {
			DRAM_access[i] *= batch;
		}
	}
	if(calc_bw) link_hops *= batch;
	return *this;
}

NoC& NoC::operator/=(const len_t& batch){
	NoC_tot_hops /= batch;
	NoP_tot_hops /= batch;
	core_hops /= batch;
	tot_DRAM_acc /= batch;
	if (!is_base) {
		for (auto i = 0; i < DRAM_num; ++i) {
			DRAM_access[i] /= batch;
		}
	}
	if(calc_bw) link_hops /= batch;
	return *this;
}

NoC::hop_t NoC::get_NoC_tot_hops() const{
	return NoC_tot_hops;
}
NoC::hop_t NoC::get_NoP_tot_hops() const {
	return NoP_tot_hops;
}


energy_t NoC::get_hop_cost() const{
	return NoC_tot_hops*NoC_hop_cost+ NoP_tot_hops * NoP_hop_cost;
}
energy_t NoC::get_NoC_hop_cost() const {
	return NoC_tot_hops * NoC_hop_cost;
}

energy_t NoC::get_NoP_hop_cost() const {

	return NoP_tot_hops * NoP_hop_cost;
}

energy_t NoC::get_tot_DRAM_cost() const {
	return DRAM_acc_cost*tot_DRAM_acc;
}

energy_t NoC::get_cost() const{
	//std::cout << "GC " << NoC_tot_hops << ' ' << tot_DRAM_acc << std::endl;
	return NoC_tot_hops * NoC_hop_cost + NoP_tot_hops * NoP_hop_cost + tot_DRAM_acc*DRAM_acc_cost;
}
// TODO: add NoC time.
cycle_t NoC::get_dram_time() const{
	//cycle_t dram_time = DIVCEIL(tot_DRAM_acc, (4*DRAM_bw));
	cycle_t dram_time = 0;
	if (!is_base) {
		for (auto access : DRAM_access) {
			if (DIVCEIL(access, DRAM_bw) > dram_time) {
				dram_time = DIVCEIL(access, DRAM_bw);
			}
		}
	}
	else {
		dram_time = DIVCEIL(tot_DRAM_acc, NoC::DRAM_num * DRAM_bw);
	}

	return dram_time;
}
access_t NoC::get_DRAM_acc(mlen_t i) const {
	//cycle_t dram_time = DIVCEIL(tot_DRAM_acc, (4*DRAM_bw));
	
	return DRAM_access[i];
}

cycle_t NoC::get_time() const{
	cycle_t dram_time = get_dram_time();
	if(!calc_bw) return dram_time;
	cycle_t nocp_time = link_hops.get_nocp_time();
	return MAX(dram_time, nocp_time);
}

void NoC::unicast(pos_t src, pos_t dst, vol_t size){
	NoC_tot_hops += unicastCalc(src, dst, size);
	hop_t temp = static_cast<hop_t>(NoP_link_calc(src, dst)) * size;
	NoP_tot_hops += temp;
	NoC_tot_hops -= temp;
}

NoC::hop_t NoC::unicastCalc(pos_t src, pos_t dst, vol_t size){
	if(calc_bw){
		noc_t x_dir = (dst.x > src.x)?0:2;
		noc_t y_dir = (dst.y > src.y)?3:1;
		mlen_t dx = (dst.x > src.x)?1:-1;
		mlen_t dy = (dst.y > src.y)?1:-1;
		for(mlen_t x = src.x; x != dst.x; x+= dx){
			link_hops.get(x, src.y, x_dir) += size;
		}
		for(mlen_t y = src.y; y != dst.y; y+= dy){
			link_hops.get(dst.x, y, y_dir) += size;
		}
	}
	return static_cast<hop_t>(abs(src.x-dst.x)+abs(src.y-dst.y)) * size;
}

void NoC::multicast(pos_t src, const pos_t* dst, cidx_t len, vol_t size){
	NoC_tot_hops += multicastCalc(src, dst, len, size);
}

NoC::hop_t NoC::multicastCalc(pos_t src, const pos_t* dst, cidx_t len, vol_t size){
	mlen_t cur_x = dst[0].x;
	mlen_t min_y = dst[0].y;
	hop_t h = 0;
	hop_t h_p_temp = 0;
	mlen_t x_temp, y_temp;
	x_temp = 0;
	y_temp = 0;
	for (cidx_t i = 1; i <= len; ++i) {
		assert(dst[i - 1].x > x_temp || (dst[i - 1].x == x_temp && dst[i - 1].y >= y_temp));
		x_temp = dst[i - 1].x;
		y_temp = dst[i - 1].y;
	}

	if(calc_bw){
		for(mlen_t x = src.x; x > dst[0].x; --x){
			link_hops.get(x, src.y, 2) += size;
		}
		for(mlen_t x = src.x; x < dst[len-1].x; ++x){
			link_hops.get(x, src.y, 0) += size;
		}
	}
	h += MAX(src.x, dst[len-1].x) - MIN(src.x, dst[0].x);
	h_p_temp = NoC::soc?0 : NoP_link_calc(MAX(src.x, dst[len - 1].x), MIN(src.x, dst[0].x));
	h -= h_p_temp;
	NoP_tot_hops += h_p_temp * size;
	for(cidx_t i=1; i<=len; ++i){
		if(i<len && dst[i].x == cur_x) continue;
		if(calc_bw){
			for(mlen_t y = src.y; y > min_y; --y){
				link_hops.get(cur_x, y, 1) += size;
			}
			for(mlen_t y = src.y; y < dst[i-1].y; ++y){
				link_hops.get(cur_x, y, 3) += size;
			}
		}
		h += MAX(src.y, dst[i-1].y) - MIN(src.y, min_y);
		h_p_temp = NoC::soc ? 0 : NoP_link_calc(MAX(src.y, dst[i - 1].y), MIN(src.y, min_y));
		h -= h_p_temp;
		NoP_tot_hops += h_p_temp * size;
		if(i == len) break;
		cur_x = dst[i].x;
		min_y = dst[i].y;
	}
	return h * size;
}

void NoC::unicast_dram(pos_t dst, vol_t size, mlen_t dram_id){
	assert(dram_id != -1);
	if (!is_base && dram_id != -2) {
		noc_t llen = dram_list[dram_id].size();
		noc_t i = 0;
		vol_t from_size = 0;
		for(const pos_t& dram: dram_list[dram_id]) {
			vol_t to_size = (size * ++i) / llen;
			unicast(dram, dst, to_size - from_size);
			from_size = to_size;
		}
		tot_DRAM_acc += size;
		DRAM_access[dram_id] += size;
	}
	else if (!is_base && dram_id == -2) {
		size = size / DRAM_num;
		for (mlen_t m = 0; m < DRAM_num; m++) {
			noc_t llen = dram_list[m].size();
			noc_t i = 0;
			vol_t from_size = 0;
			for (const pos_t& dram : dram_list[m]) {
				vol_t to_size = (size * ++i) / llen;
				unicast(dram, dst, to_size - from_size);
				from_size = to_size;
			}
			tot_DRAM_acc += size;
			DRAM_access[m] += size;
		}
	}
	else {
		noc_t llen = dram_list_base.size();
		noc_t i = 0;
		vol_t from_size = 0;
		for (const pos_t& dram : dram_list_base) {
			vol_t to_size = (size * ++i) / llen;
			unicast(dram, dst, to_size - from_size);
			from_size = to_size;
		}
		tot_DRAM_acc += size;
	}
}

void NoC::unicast_to_dram(pos_t dst, vol_t size, mlen_t dram_id){
	assert(dram_id != -1);
	if (!is_base && dram_id != -2) {
		noc_t llen = dram_list[dram_id].size();
		noc_t i = 0;
		vol_t from_size = 0;
		for(const pos_t& dram: dram_list[dram_id]) {
			vol_t to_size = (size * ++i) / llen;
			unicast(dst, dram, to_size - from_size);
			from_size = to_size;
		}
		tot_DRAM_acc += size;
		DRAM_access[dram_id] += size;
	}
	else if (!is_base && dram_id == -2) {
		size = size / DRAM_num;
		for (mlen_t m = 0; m < DRAM_num; m++) {
			noc_t llen = dram_list[m].size();
			noc_t i = 0;
			vol_t from_size = 0;
			for (const pos_t& dram : dram_list[m]) {
				vol_t to_size = (size * ++i) / llen;
				unicast(dst, dram, to_size - from_size);
				from_size = to_size;
			}
			tot_DRAM_acc += size;
			DRAM_access[m] += size;
		}
	}
	else {
		noc_t llen = dram_list_base.size();
		noc_t i = 0;
		vol_t from_size = 0;
		for (const pos_t& dram : dram_list_base) {
			vol_t to_size = (size * ++i) / llen;
			unicast(dst, dram, to_size - from_size);
			from_size = to_size;
		}
		tot_DRAM_acc += size;
	}
}

void NoC::multicast_dram(const pos_t* dst, cidx_t len, vol_t size, mlen_t dram_id){
	assert(dram_id != -1);
	if (!is_base && dram_id != -2) {
		noc_t llen = dram_list[dram_id].size();
		noc_t i = 0;
		vol_t from_size = 0;
		for(const pos_t& dram: dram_list[dram_id]) {
			vol_t to_size = (size * ++i) / llen;
			multicast(dram, dst, len, to_size - from_size);
			from_size = to_size;
		}
		tot_DRAM_acc += size;
		DRAM_access[dram_id] += size;
	}
	else if (!is_base && dram_id == -2) {
		size = size / DRAM_num;
		for (mlen_t m = 0; m < DRAM_num; m++) {
			noc_t llen = dram_list[m].size();
			noc_t i = 0;
			vol_t from_size = 0;
			for (const pos_t& dram : dram_list[m]) {
				vol_t to_size = (size * ++i) / llen;
				multicast(dram, dst, len, to_size - from_size);
				from_size = to_size;
			}
			tot_DRAM_acc += size;
			DRAM_access[m] += size;
		}
	}
	else {
		noc_t llen = dram_list_base.size();
		noc_t i = 0;
		vol_t from_size = 0;
		for (const pos_t& dram : dram_list_base) {
			vol_t to_size = (size * ++i) / llen;
			multicast(dram, dst, len, to_size - from_size);
			from_size = to_size;
		}
		tot_DRAM_acc += size;
	}
}

mlen_t NoC::NoP_link_calc(pos_t src, pos_t dst) {
	//assuming die interval is from 1 to n-1 (core id 0~n-1); n-1 core.x stands for the interval left to it. interval id = 4k,k=1~x_len/x_cut-1
	mlen_t x_max = MAX(src.x, dst.x);
	mlen_t x_min = MIN(src.x, dst.x);
	mlen_t x_NoP_hop = 0;
	mlen_t tot_NoP_hops=0;
	mlen_t y_NoP_hop = 0;
	if (x_max >= x_step) {
		mlen_t x_nearest_interval = x_max - x_max % NoC::x_step;

		assert(x_nearest_interval > 0);
		if (x_nearest_interval > x_min) {
			x_NoP_hop = 1 + (x_nearest_interval - 1 - x_min) / NoC::x_step;
		}
	}
	mlen_t y_max = MAX(src.y, dst.y);
	mlen_t y_min = MIN(src.y, dst.y);
	if (y_max >= y_step) {		
		mlen_t y_nearest_interval = y_max - y_max % NoC::y_step;
		assert(y_nearest_interval > 0);

		if (y_nearest_interval > y_min) {
			y_NoP_hop = 1 + (y_nearest_interval - 1 - y_min) / NoC::y_step;
		}
		tot_NoP_hops = x_NoP_hop + y_NoP_hop;
	}
	return tot_NoP_hops;
}
mlen_t NoC::NoP_link_calc(mlen_t x1, mlen_t x2) {
	mlen_t x_max = MAX(x1, x2);
	mlen_t x_min = MIN(x1, x2);
	mlen_t x_NoP_hop = 0;
	if (x_max >= x_step) {
		mlen_t x_nearest_interval = x_max - x_max % NoC::x_step;
		assert(x_nearest_interval > 0);
		if (x_nearest_interval > x_min) {
			x_NoP_hop = 1 + (x_nearest_interval - 1 - x_min) / NoC::x_step;
		}
	}
	return x_NoP_hop;
}
std::vector<NoC::link_info> NoC::get_link_info() const{
	std::vector<link_info> info;
	if(!calc_bw) return info;
	info.reserve(link_hops.link_hops.size());
	for(const auto& it: link_hops.link_hops){
		noc_t idx = it.first;
		noc_t dir;
		mlen_t x, y;
		if (idx >= 0) {
			dir = idx % 4;
			idx /= 4;
			y = idx % Cluster::ylen;
			idx /= Cluster::ylen;
			x = idx;
		}
		else {
			dir = idx % 4;
			x = 0;
			idx /= 4;
			y = Cluster::ylen-1+idx % Cluster::ylen;
		}
		pos_t to;
		switch(dir){
		case -3:
			to = { x, static_cast<mlen_t>(y - 1) };
			break;
		case -2:
			to = { static_cast<mlen_t>(x - 1), y };
			break;
		case -1:
			to = { x, static_cast<mlen_t>(y + 1) };
			break;
		case 0:
			to = {static_cast<mlen_t>(x+1), y};
			break;
		case 1:
			to = {x, static_cast<mlen_t>(y-1)};
			break;
		case 2:
			to = {static_cast<mlen_t>(x-1), y};
			break;
		case 3:
			to = {x, static_cast<mlen_t>(y+1)};
			break;
		default:
			assert(false);
		}
		info.push_back({{x, y}, to, it.second * link_hops.factor});
	}
	// Sort in descending.
	std::sort(info.rbegin(), info.rend());
	return info;
}

access_t NoC::get_tot_DRAM_acc() const{
	return tot_DRAM_acc;
}

vol_t NoC::calc_intersect(const fmap_range& rng1, const fmap_range& rng2, len_t bat1, len_t bat2){
	fmap_range ints = rng1.intersect(rng2);
	if(bat1 == bat2) return ints.size();

	len_t sb_st, sb_ed, lb_st, lb_ed, tot_b=0;
	if(bat1 > bat2){
		assert(bat1 % bat2 == 0);
		sb_st = rng2.b.from;
		sb_ed = rng2.b.to;
		lb_st = rng1.b.from;
		lb_ed = rng1.b.to;
		for(;sb_st < lb_ed; sb_st+=bat2, sb_ed+=bat2){
			if(sb_ed <= lb_st) continue;
			tot_b += MIN(sb_ed, lb_ed) - MAX(sb_st, lb_st);
		}
	}else{
		assert(bat2 % bat1 == 0);
		sb_st = rng1.b.from;
		sb_ed = rng1.b.to;
		lb_st = rng2.b.from;
		lb_ed = rng2.b.to;
		for(;sb_st < lb_ed; sb_st+=bat1, sb_ed+=bat1){
			if(sb_ed <= lb_st) continue;
			tot_b += MIN(sb_ed, lb_ed) - MAX(sb_st, lb_st);
		}
	}
	ints.b.from=0;
	ints.b.to=tot_b;
	return ints.size();
}

std::ostream& operator<<(std::ostream& os, const NoC& noc){
	return os<<" NoC(NoC_tot_hops= "<<noc.NoC_tot_hops << " NoP_tot_hops= " << noc.NoP_tot_hops <<",core_hops="<<noc.core_hops<<", DRAM acc="<<noc.tot_DRAM_acc<<")";
}

NoC::HopCount::HopCount():factor(1){
}

NoC::HopCount& NoC::HopCount::operator+=(const HopCount& other){
	if(factor > 1){
		for(const auto& it: link_hops){
			link_hops[it.first] *= factor;
		}
	}
	for(const auto& it : other.link_hops){
		link_hops[it.first] += it.second * other.factor;
	}
	return *this;
}

NoC::HopCount& NoC::HopCount::operator*=(const len_t& batch){
	factor *= batch;
	return *this;
}

NoC::HopCount& NoC::HopCount::operator/=(const len_t& batch){
	if(factor % batch == 0){
		factor /= batch;
	}else{
		for(const auto& it: link_hops){
			link_hops[it.first] = (it.second * factor) / batch;
		}
		factor = 1;
	}
	return *this;
}

NoC::hop_t& NoC::HopCount::get(mlen_t x, mlen_t y, mlen_t dir){
	noc_t idx = (x * Cluster::ylen + y) * 4 + dir;
	assert(idx < 4*(Cluster::xlen+1)*Cluster::ylen);
	return link_hops[idx];
}

void NoC::HopCount::clear(){
	link_hops.clear();
}


/*
NoC::hop_t NoC::HopCount::max() const {
	hop_t h = 0;
	for (const auto& it : link_hops) {
		h = MAX(h, it.second);
	}
	return h * factor;
}*/

NoC::hop_t NoC::HopCount::max() const{
	hop_t h_noc = 0;
	hop_t h_nop = 0;
	for(const auto& it: link_hops){
		if (NoP_links.count(it.first)) {
			h_nop = MAX(h_nop, it.second)*NoC_bw/NoP_bw;
		}
		else {
			h_noc = MAX(h_noc, it.second);
		}
	}
	return MAX(h_noc,h_nop) * factor;
}

cycle_t NoC::HopCount::get_nocp_time() const {
	hop_t NoC = 0;
	hop_t NoP = 0;
	for (const auto& it : link_hops) {
		if (NoP_links.count(it.first)) {
			NoP= MAX(NoP, it.second);
		}
		else {
			NoC = MAX(NoC, it.second);
		}
	}
	return MAX(DIVCEIL(NoP*factor,NoP_bw), DIVCEIL(NoC * factor, NoC_bw));
}

bool NoC::link_info::operator<(const link_info& other) const{
	if(total_hops != other.total_hops) return total_hops < other.total_hops;
	if(from != other.from) return from < other.from;
	return to < other.to;
}

bool NoC::link_info::operator==(const link_info& other) const{
	return total_hops == other.total_hops && from == other.from && to == other.to;
}

bool NoC::link_info::operator>(const link_info& other) const{
	if(total_hops != other.total_hops) return total_hops > other.total_hops;
	if(from != other.from) return from > other.from;
	return to > other.to;
}

std::ostream& operator<<(std::ostream& os, const NoC::link_info& info){
	return os<<info.from<<" -> "<<info.to<<'\t'<<info.total_hops;
}


