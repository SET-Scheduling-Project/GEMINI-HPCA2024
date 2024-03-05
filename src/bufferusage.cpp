#include "bufferusage.h"

#include <cassert>
#include "layerengine.h"
#include "schnode.h"

BufferUsage::BufferUsage()
	:BufferUsage(SchNode::layerMapper->get_ubuf_size()){}

BufferUsage::BufferUsage(vol_t _max_vol)
	:factor(1),max_vol(_max_vol),valid(true){}

bool BufferUsage::add(pos_t chip, vol_t size){
	if(factor == 1){
		valid &= (usage[chip] += size) <= max_vol;
	}else if(size % factor == 0){
		valid &= (usage[chip] += size / factor)*factor <= max_vol;
	}else{
		for(auto x:usage){
			usage[x.first] *= factor;
		}
		factor = 1;
		valid &= (usage[chip] += size) <= max_vol;
	}
	return valid;
}

bool BufferUsage::all_add(vol_t size){
	for(auto x:usage){
		valid &= (usage[x.first] += size) <= max_vol;
	}
	return valid;
}

BufferUsage& BufferUsage::operator+=(const BufferUsage& other){
	assert(&other != this);
	if(!valid || !other.valid){
		valid = false;
		return *this;
	}
	if(factor != 1){
		for(auto x:usage){
			usage[x.first] *= factor;
		}
		factor = 1;
	}
	for(auto x:other.usage){
		valid &= (usage[x.first] += x.second * other.factor) <= max_vol;
	}
	return *this;
}

BufferUsage BufferUsage::operator+(const BufferUsage& other) const{
	BufferUsage u = other;
	u += *this;
	return u;
}

void BufferUsage::max(const BufferUsage& other){
	if(!other.valid || !valid){
		valid = false;
		return;
	}
	if(factor == 1){
		for(auto x:other.usage){
			usage[x.first] = MAX(usage[x.first], x.second);
		}
	}else{
		// TODO:impl here.
		assert(false);
	}
}

vol_t BufferUsage::max() const{
	if(!valid) return 0;
	vol_t max_vol = 0;
	for(auto x:usage){
		max_vol = MAX(max_vol, x.second);
	}
	return max_vol;
}

double BufferUsage::avg() const{
	if(!valid) return 0;
	if(usage.empty()) return 0;
	double avg_vol = 0;
	for(auto x:usage){
		avg_vol += x.second;
	}
	return avg_vol / usage.size();
}

bool BufferUsage::multiple(vol_t n){
	for(auto x:usage){
		valid &= (usage[x.first] *= n) <= max_vol;
	}
	return valid;
}

vol_t BufferUsage::get_max_vol() const{
	return max_vol;
}

std::ostream& operator<<(std::ostream& os, const BufferUsage& usage){
	return os<<"Buffer(max="<<usage.max()<<", avg="<<usage.avg()<<")";
}

BufferUsage::operator bool() const{
	return valid;
}
