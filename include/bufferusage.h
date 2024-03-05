#ifndef BUFFERUSAGE_H
#define BUFFERUSAGE_H

#include <unordered_map>
#include "util.h"

class BufferUsage{
	struct pos_hash {
		std::size_t operator()(const pos_t& pos) const {
			return std::hash<pos_t::pos_hash_t>{}(*reinterpret_cast<const pos_t::pos_hash_t*>(&pos));
		}
	};
	std::unordered_map<pos_t, vol_t, pos_hash> usage;
	vol_t factor, max_vol;
	bool valid;
public:
	BufferUsage();
	BufferUsage(vol_t _max_vol);
	bool add(pos_t chip, vol_t size);
	bool all_add(vol_t size);
	operator bool() const;
	BufferUsage& operator+=(const BufferUsage& other);
	BufferUsage operator+(const BufferUsage& other) const;
	void max(const BufferUsage& other);
	vol_t max() const;
	double avg() const;
	bool multiple(vol_t n);
	vol_t get_max_vol() const;

	friend std::ostream& operator<<(std::ostream& os, const BufferUsage& usage);
};

#endif // BUFFERUSAGE_H
