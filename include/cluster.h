#ifndef CLUSTER_H
#define CLUSTER_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "util.h"

class Cluster{
public:
	/*
	struct core{
		pos_t pos;
		vol_t UBUF_vol;
	};
	*/
	typedef vol_t hop_t;
	typedef cidx_t xyid_t;
	typedef std::unique_ptr<cidx_t[]> allocRes_t;
	static mlen_t xlen,ylen,stride;
	static double min_util;
private:
	//const core* cores;
	struct{
		cidx_t first, last;
	}range;

	// TODO: use core_list in the future,
	//       when cores are not allocated in rectangular fashion.
	bool use_range;
	

public:
	Cluster(cidx_t _first, cidx_t _last);
	Cluster(std::vector<cidx_t>& _tilelist);//for spatial mapping
	bool operator==(const Cluster& other) const;
	bool operator!=(const Cluster& other) const;

	void erase(std::vector<cidx_t> & _erase_list);//for spatial mapping
	void insert(std::vector<cidx_t>& _insert_loc, std::vector<cidx_t>& _insert_value);//for spatial mapping
	void erase(cidx_t _erase_element);//for spatial mapping
	void insert(cidx_t _insert_loc, cidx_t _insert_value);//for spatial mapping
	void change_to_list();//change range mode to list mode

	cidx_t num_cores() const;
	allocRes_t try_alloc(utime_t* ops, cidx_t childNum, utime_t totOps=0, bool base= false) const;
	Cluster sub_cluster(cidx_t childIdx, const allocRes_t& allocRes) const;
	Cluster sub_cluster(cidx_t from, cidx_t num) const;
	Cluster sub_cluster(std::vector<cidx_t>& _id_list) const;//for spatial mapping
	void set_core_set();
	std::unordered_set<cidx_t> core_set;
	std::vector<cidx_t> core_list;
	
	/*
	static hop_t unicast(pos_t src, pos_t dst);
	// TODO: dst needs to be in inc. order.
	static hop_t multicast(pos_t src, pos_t* dst, cidx_t len);
	// DRAM is at (-1,x) and (n,x)
	static hop_t unicast_dram(pos_t dst, vol_t size);
	static hop_t unicast_to_dram(pos_t dst, vol_t size);
	static hop_t multicast_dram(pos_t* dst, cidx_t len, vol_t size);
	*/
	static pos_t get_pos(cidx_t core_idx);
	static xyid_t get_xyid(pos_t& chip);
	// bool first_chip(pos_t& chip) const;
	// bool next_chip(pos_t& chip) const;
	pos_t operator[](cidx_t num_chip) const;
	memidx_t nearest_dram();
};


#endif // CLUSTER_H
