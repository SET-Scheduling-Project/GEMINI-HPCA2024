#include "placement.h"
#include "spatial_mapping/light_placement.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include "cluster.h"
#include "network.h"

PlaceEngine placeEngine;
/*
PlaceSch::PlaceSch(len_t _batch):batch_size(_batch){
}
*/

PlaceSch::PlaceSch(const PlaceSch& sch):part(sch.part), fetch(sch.fetch),
	ifmLayout(sch.ifmLayout->clone()),
	wgtLayout(sch.wgtLayout->clone()),
	ofmLayout(sch.ofmLayout->clone()){
	memcpy(order, sch.order, sizeof(order[0])*4);
}

void PlaceSch::finalize(){
	ifmLayout->finalize();
	wgtLayout->finalize();
	ofmLayout->finalize();
	permuteOrder.reset();
}

void PlaceSch::initPlacement_by_light(const Cluster& cluster, const Light_placement& place, lid_t layerno){
	//TODO
	using plen_t = PartSch::partlen_t;

	pos_t* curIdx = permuteOrder.get();

	order[0] = 1;
	order[1] = 0;
	order[2] = 2;
	order[3] = 3;

	cidx_t idx = 0;
	for(auto core: place.get_placement()){
		if(core.layerno==layerno){
			*(curIdx++) = cluster[idx++];
		}
	}
}

void PlaceSch::initPlacement(const Cluster& cluster){
	using plen_t = PartSch::partlen_t;

	pos_t* curIdx = permuteOrder.get();

	plen_t step[4] = {1,1,1,1};
	step[order[3]] = 1;
	step[order[2]] = part[order[3]];
	step[order[1]] = part[order[3]] * part[order[2]];
	step[order[0]] = part[order[3]] * part[order[2]] * part[order[1]];
	for(plen_t k = 0; k < part.K; ++k){
		for(plen_t b = 0; b < part.B; ++b){
			for(plen_t h = 0; h < part.H; ++h){
				for(plen_t w = 0; w < part.W; ++w){
					// Init (b, k, h, w)
					plen_t idx = step[0] * k + step[1] * b + step[2] * h + step[3] * w;
					*(curIdx++) = cluster[idx];
				}
			}
		}
	}
}

void PlaceSch::update(PlaceSch&& sch){
	part = sch.part;
	fetch = sch.fetch;
	memcpy(order, sch.order, sizeof(order[0])*4);
	//ifmLayout = std::move(sch.ifmLayout);
	//wgtLayout = std::move(sch.wgtLayout);
	//ofmLayout = std::move(sch.ofmLayout);
	//permuteOrder = std::move(sch.permuteOrder);
}

DataLayout& PlaceSch::getIfmL(){
	return *ifmLayout.get();
}

const DataLayout& PlaceSch::getIfmL() const{
	return *ifmLayout.get();
}

DataLayout& PlaceSch::getWgtL(){
	return *wgtLayout.get();
}

const DataLayout& PlaceSch::getWgtL() const{
	return *wgtLayout.get();
}

UniqueLayout& PlaceSch::getOfmL(){
	return *ofmLayout.get();
}

const UniqueLayout& PlaceSch::getOfmL() const{
	return *ofmLayout.get();
}

std::ostream& operator<<(std::ostream& os, const PlaceSch& sch){
	os << '(';
	for(int i=0;i<4;++i){
		switch(sch.order[i]){
		case 0: os << "K:" << sch.part.K;
			break;
		case 1: os << "B:" << sch.part.B;
			break;
		case 2: os << "H:" << sch.part.H;
			break;
		case 3: os << "W:" << sch.part.W;
			break;
		default: os << "X:X";
			break;
		}
		if(i != 3) os << ',';
	}
	return os << ')';
}

PlaceIter PlaceEngine::init(PlaceSch& cur_sch){
	return PlaceIter(cur_sch);
}

PlaceIter::PlaceIter(PlaceSch& placeSch) : curSch(placeSch){
	std::uint8_t first = 0, last = 3;
	for(std::uint8_t i=0; i<4; ++i){
		if(curSch.part[i] == 1){
			curSch.order[last--]=i;
		}else{
			curSch.order[first++]=i;
		}
	}
	++last;
	assert(first == last);
	perm_len = first;
	hasNext = true;
}

bool PlaceIter::nextPlace(cost_t cost){
	(void) cost;
	return hasNext = std::next_permutation(curSch.order, curSch.order+perm_len);
}

PlaceIter::operator bool() const{
	return hasNext;
}

fmap_range range_from_partition_number(const fmap_shape &shape, len_t batch, const PartSch &partition, cidx_t id){
	len_t b_len = DIVCEIL(batch,partition.B);
	len_t k_len = DIVCEIL(shape.c,partition.K);
	len_t h_len = DIVCEIL(shape.h,partition.H);
	len_t w_len = DIVCEIL(shape.w,partition.W);
	len_t* arrs[4];
	arrs[0] = part_intv(shape.c, partition.K);
	arrs[1] = part_intv(batch, partition.B);
	arrs[2] = part_intv(shape.h, partition.H);
	arrs[3] = part_intv(shape.w, partition.W);
	len_t w_id = id%partition.W;
	id /= partition.W;
	len_t h_id = id%partition.H;
	id /= partition.H;
	len_t c_id = id%partition.K;
	id /= partition.K;
	len_t b_id = id;
	auto ret = fmap_range{
		{arrs[0][c_id], arrs[0][c_id+1]},
		{arrs[1][b_id], arrs[1][b_id+1]},
		{arrs[2][h_id], arrs[2][h_id+1]},
		{arrs[3][w_id], arrs[3][w_id+1]}
	};
	delete[] arrs[0];
	delete[] arrs[1];
	delete[] arrs[2];
	delete[] arrs[3];
	return ret;
}
