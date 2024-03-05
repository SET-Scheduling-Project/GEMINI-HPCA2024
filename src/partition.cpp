#include "partition.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include "network.h"

#define MAX_BUF MAX_CHIPS

PartEngine::fvec PartEngine::factors[MAX_BUF+1];
double PartEngine::utils[MAX_BUF+1][MAX_BUF+1];

PartEngine partEngine;

std::vector<PartEngine::num_pair> PartEngine::factor_num(PartEngine::factor_t n){
	std::vector<PartEngine::num_pair> f;
	if(n <= 11){
		switch (n) {
		case 1: f={{1,1}}; break;
		case 4: f={{2,2}, {1,4}, {4,1}}; break;
		case 6: f={{2,3}, {3,2}, {1,6}, {6,1}}; break;
		case 8: f={{2,4}, {4,2}, {1,8}, {8,1}}; break;
		case 9: f={{3,3}, {1,9}, {9,1}}; break;
		case 10:f={{2,5}, {5,2}, {1,10}, {10,1}}; break;
		default:
			f={{1,n}, {n,1}};
		}
		return f;
	}
	factor_t max_t = static_cast<factor_t>(std::sqrt(n+0.5));
	if(n%max_t == 0){
		f.push_back({max_t,static_cast<factor_t>(n/max_t)});
		if(max_t*max_t!=n) f.push_back({static_cast<factor_t>(n/max_t), max_t});
	}
	for(factor_t i=max_t-1; i>1; --i){
		if(n%i==0){
			f.push_back({i,static_cast<factor_t>(n/i)});
			f.push_back({static_cast<factor_t>(n/i),i});
		}
	}
	f.push_back({1,n});
	f.push_back({n,1});
	return f;
}

void PartEngine::init_all(){
	// Init factors.
	factors[0].push_back({0,0,0,0});
	for(factor_t i=1; i<=MAX_BUF; ++i){
		for(auto xy: factor_num(i)){
			for(auto pq: factor_num(xy.x)){
				for(auto rs: factor_num(xy.y)){
					factors[i].push_back({pq.x, pq.y, rs.x, rs.y});
				}
			}
		}
	}
	// Init utils.
	for(factor_t i=1; i<=MAX_BUF; ++i){
		for(factor_t j=1; j<=MAX_BUF; ++j){
			double d = i;
			d /= DIVCEIL(i, j) * j;
			utils[i][j]=d;
		}
	}
}

PartEngine::PartEngine(double _min_util):min_util(_min_util){
	if(factors[0].empty()){
		init_all();
	}
}

PartIter PartEngine::init(cidx_t cluster_size, len_t batch_num, const Node& layer, PartSch& sch, len_t min_cuts){
	if(cluster_size > MAX_BUF){
		// TODO: add support for more cores.
		assert(false);
	}
	PartIter it(sch, min_util);
	it.min_ncut = min_cuts;
	it.endPos = factors[cluster_size].end();
	it.nextPos = factors[cluster_size].begin();
	it.maxB = batch_num;
	const auto& ofm_shape = layer.layer().ofmap_shape();
	it.maxK = ofm_shape.c;
	it.maxH = ofm_shape.h;
	it.maxW = ofm_shape.w;
	if(!it.nextPart()){
		it.nextPos = factors[cluster_size].begin();
		it.finished = true;
		//it.finished = !it.getBestPart();
	}
	return it;
}

bool PartIter::calcUtil(const PartSch& nextSch) const{
	if(nextSch.B * nextSch.H * nextSch.W < min_ncut) return false;
	double util = calc_util(maxK, nextSch.K)\
				* calc_util(maxB, nextSch.B)\
				* calc_util(maxH, nextSch.H)\
				* calc_util(maxW, nextSch.W);
	return util >= min_util;
}

bool PartIter::calc_util_best(const PartSch& nextSch){
	if(nextSch.B * nextSch.H * nextSch.W < min_ncut) return false;
	double util = calc_util(maxK, nextSch.K)\
				* calc_util(maxB, nextSch.B)\
				* calc_util(maxH, nextSch.H)\
				* calc_util(maxW, nextSch.W);
	if(util <= min_util) return false;
	min_util = util;
	return true;
}

PartIter::PartIter(PartSch& partSch, double _min_util)
	:curSch(partSch), min_util(_min_util), finished(false){}

double PartIter::calc_util(len_t real, len_t part){
	if(real <= MAX_BUF) return PartEngine::utils[real][part];
	return static_cast<double>(real)/(DIVCEIL(real, part) * part);
}

bool PartIter::nextPart(cost_t cost){
	(void) cost;
	while (nextPos != endPos) {
		const PartSch& nextSch = *(nextPos++);
		if(nextSch.B<=maxB && nextSch.K<=maxK && nextSch.H<=maxH && nextSch.W<=maxW &&calcUtil(nextSch)){
			curSch = nextSch;
			return true;
		}
	}
	finished = true;
	return false;
}

PartIter::operator bool() const{
	return !finished;
}

bool PartIter::getBestPart(cost_t cost){
	(void) cost;
	bool p = false;
	double u = min_util;
	min_util = 0;
	while (nextPos!= endPos) {
		auto& nextSch = *(nextPos++);
		if(calc_util_best(nextSch)){
			curSch = nextSch;
			p = true;
		}
	}
	min_util = u;
	nextPos = endPos;
	return p;
}

PartSch::PartSch(len_t _K, len_t _B, len_t _H, len_t _W)
	:K(_K), B(_B), H(_H), W(_W){}

len_t& PartSch::operator[](std::uint8_t i){
	/*switch (i) {
	case 0: return K;
	case 1: return B;
	case 2: return H;
	case 3: return W;
	default: break;
	}
	assert(false);
	return B;
	*/
	return reinterpret_cast<len_t*>(this)[i];
}


const len_t& PartSch::operator[](std::uint8_t i) const{
	/*switch (i) {
	case 0: return K;
	case 1: return B;
	case 2: return H;
	case 3: return W;
	default: break;
	}
	assert(false);
	return B;
	*/
	return reinterpret_cast<const len_t*>(this)[i];
}

vol_t PartSch::size() const{
	return K*B*H*W;
}

std::ostream& operator<<(std::ostream& os, const PartSch& sch){
	return os << "(B=" << sch.B << ",K=" << sch.K << ",H=" << sch.H << ",W=" << sch.W << ')';
}
FetchSch::FetchSch(len_t _K, len_t _B, len_t _H, len_t _W, len_t _ifmF, len_t _wgtF)
	:PartSch(_K, _B, _H, _W), ifmFetch(_ifmF), wgtFetch(_wgtF) {}

FetchSch::operator bool() const {
	return B != 0;
}

void FetchSch::clear() {
	B = 0;
	H = 0;
	K = 0;
	W = 0;
	ifmFetch = 1;
	wgtFetch = 1;
}