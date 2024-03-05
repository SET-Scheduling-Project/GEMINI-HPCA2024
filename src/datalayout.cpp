#include "datalayout.h"

#include "bufferusage.h"
#include "layer.h"
#include "partition.h"
#include <algorithm>
#include <cassert>
#include <cstring>

void DataLayout::update(fmap_range& range){
	vol_t s = range.size();
	totVolume += s * bcastLength();
	if (s > maxVolume) {
		maxVolume =  s;
		MaxRange = range;
	}
}

DataLayout::DataLayout():totVolume(0), maxVolume(0), multFactor(1){}

vol_t DataLayout::totalSize() const{
	return totVolume;
}

vol_t DataLayout::maxRange() const{
	return maxVolume;
}

fmap_range DataLayout::MaxRange_() const {
	return MaxRange;
}

void DataLayout::sizeMult(len_t num){
	multFactor *= num;
	totVolume *= num;
	maxVolume *= num;
}

void DataLayout::clear(){
	totVolume = maxVolume = 0;
	multFactor = 1;
}

bool DataLayout::update(BufferUsage& usage, const Layer& l, const FetchSch& fetch, bool isIfm) const {
	dataLen_t totL = totLength();
	for (dataLen_t i = 0; i < totL; ++i) {
		UniqueEntry ent = (*this)[i];
		fmap_range r = ent.range;
		vol_t s = r.size();
		if (s == 0) continue;
		if (fetch) {
			if (isIfm) {
				s = l.ifm_part(r, fetch);
			}
			else {
				s = l.wgt_part(r, fetch);
			}
		}
		s *= multFactor;
		if (!usage.add(ent.tile, s)) return false;
	}
	return true;
}

DataLayout::dataLen_t DataLayout::bcastLength() const{
	return totLength() / rangeLength();
}

UniqueLayout::UniqueLayout(dataLen_t _len):len(_len){}

UniqueLayout::Iterator UniqueLayout::begin() const{
	return Iterator(*this);
}

UniqueLayout::Iterator UniqueLayout::end() const{
	return Iterator(*this, len);
}

DataLayout::dataLen_t UniqueLayout::totLength() const{
	return len;
}

DataLayout::dataLen_t UniqueLayout::rangeLength() const{
	return len;
}

DataLayout::Entry UniqueLayout::at(dataLen_t idx) const{
	UniqueEntry u = (*this)[idx];
	return {u.range, &u.tile, 1, u.divN};
}

StdDataLayout::StdDataLayout(dataLen_t _len, pos_t* _posArr)
	:range_len(_len), bcast_len(1), tot_len(_len), contPosArr((_len>0)?std::make_unique<pos_t[]>(_len):nullptr), posArr(_posArr){}

DataLayout* StdDataLayout::clone() const{
	StdDataLayout* newLayout = new StdDataLayout(tot_len, nullptr);
	if(tot_len <= 0) return newLayout;
	newLayout->range_len = range_len;
	newLayout->bcast_len = bcast_len;
	newLayout->bcast_down = bcast_down;
	newLayout->bcast_step = bcast_step;
	newLayout->rangeArr = std::make_unique<fmap_range[]>(range_len);
	newLayout->posArr = newLayout->contPosArr.get();
	memcpy(newLayout->rangeArr.get(), rangeArr.get(), sizeof(rangeArr[0]) * range_len);
	memcpy(newLayout->posArr, posArr, sizeof(posArr[0]) * tot_len);
	newLayout->totVolume = totVolume;
	newLayout->maxVolume = maxVolume;
	newLayout->multFactor = multFactor;
	return newLayout;
}

void StdDataLayout::finalize(){
	if(tot_len <= 0){
		posArr = nullptr;
		return;
	}
	if(bcast_len == 1){
		memcpy(contPosArr.get(), posArr, sizeof(posArr[0]) * tot_len);
	}
	posArr = contPosArr.get();
}

void StdDataLayout::reset(){
	range_len = bcast_len = tot_len = bcast_step = bcast_down = 0;
	contPosArr.reset();
	rangeArr.reset();
	posArr = nullptr;
}

DataLayout::dataLen_t StdDataLayout::totLength() const{
	return tot_len;
}

DataLayout::dataLen_t StdDataLayout::rangeLength() const{
	return range_len;
}

DataLayout::Entry StdDataLayout::at(dataLen_t idx) const{
	if(bcast_len == 1) return {rangeArr[idx], posArr + idx, 1, 1};
	return {rangeArr[idx], &contPosArr[idx*bcast_len], bcast_len, 1};
}

DataLayout::UniqueEntry StdDataLayout::operator[](dataLen_t idx) const{
	if(bcast_len == 1) return {rangeArr[idx], posArr[idx], 1};
	return {rangeArr[idx/bcast_len], contPosArr[idx], 1};
}

void StdDataLayout::setCPosArr(){
	if(bcast_len == 1) return;
	if(bcast_step == 1){
		for(dataLen_t i = 0; i < tot_len; i += bcast_len){
			memcpy(&contPosArr[i], posArr+i, sizeof(posArr[0])*bcast_len);
			std::sort(&contPosArr[i], &contPosArr[i+bcast_len]);
		}
		return;
	}
	for(dataLen_t i = 0; i < tot_len; i += bcast_down){
		for(dataLen_t j = 0; j < bcast_step; ++j){
			for(dataLen_t k = 0; k < bcast_len; ++k){
				contPosArr[i+j*bcast_len+k] = posArr[i+j+k*bcast_step];
			}
			std::sort(&contPosArr[i+j*bcast_len], &contPosArr[i+(j+1)*bcast_len]);
		}
	}
}

void StdDataLayout::setBcast(dataLen_t _bcastLen, dataLen_t _bcastStep){
	bcast_len = _bcastLen;
	range_len = tot_len / bcast_len;
	bcast_step = _bcastStep;
	bcast_down = _bcastStep * _bcastLen;
	rangeArr.reset(new fmap_range[range_len]);
	clear();
}

StdULayout::StdULayout(dataLen_t _len, pos_t* _posArr)
	:UniqueLayout(_len), rangeArr((_len>0)?std::make_unique<fmap_range[]>(_len):nullptr), posArr(_posArr){}

UniqueLayout* StdULayout::clone() const{
	StdULayout* newLayout = new StdULayout(len, nullptr);
	if(len <= 0) return newLayout;
	memcpy(newLayout->dimLen, dimLen, sizeof(dimLen[0]) * 4);
	memcpy(newLayout->dimStep, dimStep, sizeof(dimStep[0]) * 4);
	memcpy(newLayout->rangeArr.get(), rangeArr.get(), sizeof(rangeArr[0]) * len);
	newLayout->localPosArr = std::make_unique<pos_t[]>(len);
	newLayout->posArr = newLayout->localPosArr.get();
	memcpy(newLayout->posArr, posArr, sizeof(posArr[0]) * len);
	newLayout->totVolume = totVolume;
	newLayout->maxVolume = maxVolume;
	newLayout->multFactor = multFactor;
	return newLayout;
}

void StdULayout::finalize(){
	if(len <= 0){
		posArr = nullptr;
		return;
	}
	localPosArr = std::make_unique<pos_t[]>(len);
	memcpy(localPosArr.get(), posArr, sizeof(posArr[0])*len);
	posArr = localPosArr.get();
}

void StdULayout::reset(){
	len = 0;
	memset(dimLen, 0, sizeof(dimLen[0])*4);
	rangeArr.reset();
	posArr = nullptr;
}

void StdULayout::setDims(dataLen_t C, dataLen_t B, dataLen_t H, dataLen_t W){
	dimLen[0] = C;
	dimLen[1] = B;
	dimLen[2] = H;
	dimLen[3] = W;
	dimStep[0] = B*H*W;
	dimStep[1] = H*W;
	dimStep[2] = W;
	dimStep[3] = 1;
	clear();
}

DataLayout::UniqueEntry StdULayout::operator[](dataLen_t idx) const{
	return {rangeArr[idx], posArr[idx], 1};
}

std::vector<std::pair<fmap_range, pos_t> > StdULayout::get_intersect_bruteforce(const fmap_range& range, len_t batch_size, len_t range_batch_size) const{
	std::vector<std::pair<fmap_range, pos_t> > ret;
	for(auto it: *this){
		len_t gcd = std::min(batch_size, range_batch_size);
		assert(batch_size%gcd==0);
		assert(range_batch_size%gcd==0);
		bool intersect = false;
		for(len_t i=0; i<batch_size/gcd; ++i){
			for(len_t j=0; j<range_batch_size/gcd; ++j){
				auto range_offset = range;
				range_offset.b += j*gcd;
				auto offset = it.first;
				offset.b += i*gcd;
				if(!offset.intersect(range_offset).is_empty()){
					intersect = true;
				}
			}
		}
		if(intersect){
			ret.push_back(it);
		}
	}
	return ret;
}

StdULayout::IntersectIter StdULayout::get_intersect(const fmap_range& range, bool noBatch) const{
	dataLen_t from[4];
	dataLen_t to[4];
	for(int i=0; i<4; ++i){
		if(noBatch && i == 1){
			from[i] = 0;
			to[i] = dimLen[i];
			continue;
		}
		dataLen_t j=0;
		len_t rFrom = range.get_range(i).from;
		while(j<dimLen[i] && rangeArr[j*dimStep[i]].get_range(i).to <= rFrom) ++j;
		from[i] = j;
		if(j == dimLen[i]){
			to[0] = from[0];
			break;
		}
		len_t rTo = range.get_range(i).to;
		while(j<dimLen[i] && rangeArr[j*dimStep[i]].get_range(i).from < rTo) ++j;
		to[i] = j;
	}
	return IntersectIter(from, to, *this);
}
/*
MemULayout::MemULayout(const fmap_range& _range, pos_t* _posArr, len_t memLen)
	:UniqueLayout(memLen), range(_range), posArr(_posArr){}

void MemULayout::reset(){
	len = 0;
	posArr = nullptr;
}

DataLayout::UniqueEntry MemULayout::operator[](dataLen_t idx) const{
	return {range, posArr[idx], len};
}
*/
UniqueLayout::Iterator::Iterator(const UniqueLayout& _layout, dataLen_t _i)
	:i(_i), layout(_layout){}

UniqueLayout::Iterator& UniqueLayout::Iterator::operator++(){
	++i;
	return *this;
}

std::pair<fmap_range, pos_t> UniqueLayout::Iterator::operator*() const{
	UniqueEntry entry = layout[i];
	return {entry.range, entry.tile};
}

bool UniqueLayout::Iterator::operator!=(const Iterator& other) const{
	return &layout != &other.layout || i != other.i;
}

StdULayout::IntersectIter::IntersectIter(dataLen_t _from[], dataLen_t _to[], const StdULayout& _layout)
	:layout(_layout){
	memcpy(from, _from, sizeof(from[0])*4);
	memcpy(to, _to, sizeof(to[0])*4);
	memcpy(cur, _from, sizeof(cur[0])*4);
}

DataLayout::UniqueEntry StdULayout::IntersectIter::operator*() const{
	DataLayout::dataLen_t idx = 0;
	for(int i=0; i<4; ++i)
		idx += cur[i] * layout.dimStep[i];
	return layout[idx];
}

bool StdULayout::IntersectIter::isValid() const{
	return to[0] > from[0];
}

void StdULayout::IntersectIter::next(){
	if(++cur[3] >= to[3]){
		cur[3] = from[3];
		if(++cur[2] >= to[2]){
			cur[2] = from[2];
			if(++cur[1] >= to[1]){
				cur[1] = from[1];
				if(++cur[0] >= to[0]){
					from[0] = to[0];
				}
			}
		}
	}
}
