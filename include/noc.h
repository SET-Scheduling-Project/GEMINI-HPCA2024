#ifndef NOC_H
#define NOC_H

#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "cluster.h"
#include "util.h"

class DataLayout;
class UniqueLayout;
struct PlaceSch;
class LNode;
class BufferUsage;
//#include "datalayout.h"
//#include "placement.h"
//#include "schnode.h"
//#include "bufferusage.h"

class NoC{
public:
	typedef Cluster::hop_t hop_t;
	static energy_t NoC_hop_cost,NoP_hop_cost, DRAM_acc_cost;
	static bw_t DRAM_bw, NoC_bw, NoP_bw;
	static bool interleave;
	static bool soc;
	static bool seperate_IO;//whether IO chiplet is seperate
	static bool calc_noc_control;
	static mlen_t DRAM_num,DRAM_router_num,x_cut,y_cut,x_step,y_step;
	//x,y step is core number of each chiplet
	static std::vector<std::vector<pos_t>> dram_list;
	static std::vector<pos_t> dram_list_base;
	//static 
	static std::unordered_set<noc_t>NoP_links;
	bool calc_bw;
	bool is_base;
private:
	mlen_t NoP_link_calc(pos_t src, pos_t dst);
	mlen_t NoP_link_calc(mlen_t x1, mlen_t x2);
	class HopCount{
		// TODO: change size_t to appropriate size
		std::unordered_map<noc_t, hop_t> link_hops;
		//std::unordered_set<noc_t> NoP_links;
		hop_t factor;
	public:
		HopCount();
		HopCount& operator+=(const HopCount& other);
		HopCount& operator*=(const len_t& batch);
		HopCount& operator/=(const len_t& batch);
		hop_t& get(mlen_t x, mlen_t y, mlen_t dir);
		//std::unordered_set<noc_t>& get_NoP();
		void clear();
		hop_t max() const;
		cycle_t  get_nocp_time() const;
		friend class NoC;
	};

	// TODO: handle calc_bw=true;
	
	// TODO: get rid of all static_cast
	// (may need to change PartSch to cidx_t)

	hop_t NoC_tot_hops;
	hop_t NoP_tot_hops;
	hop_t core_hops;
	access_t tot_DRAM_acc;
	std::vector<access_t>DRAM_access;
	// Direction: ESWN = 0123
	HopCount link_hops;
	vol_t calc_intersect(const fmap_range& rng1, const fmap_range& rng2, len_t bat1, len_t bat2);
public:
	NoC(bool _calc_bw = true, bool base = false);
	//static std::unordered_set<size_t> initial_NoP_links();
	NoC(const NoC& other) = default;
	NoC(NoC&& other) = default;
	NoC& operator=(const NoC& other) = default;
	NoC& operator=(NoC&& other) = default;

	NoC operator+(const NoC& other) const;
	NoC& operator+=(const NoC& other);
	NoC operator*(const len_t& batch) const;
	NoC& operator*=(const len_t& batch);
	NoC& operator/=(const len_t& batch);
	~NoC() = default;

	//void set_calc_bw(bool _calc_bw);
	void reset();

	void fromRemoteMem(const DataLayout& toLayout,mlen_t dram_id);
	void fromRemoteMem(const DataLayout& toLayout, len_t fromC, len_t toC, mlen_t dram_id);
	void toRemoteMem(const UniqueLayout& fromLayout, mlen_t dram_id);
	void betweenLayout(const UniqueLayout& fromLayout, const DataLayout& toLayout, len_t fromCOffset, len_t fromB, len_t toB, bool unordered = false);

	hop_t get_NoC_tot_hops() const;
	hop_t get_NoP_tot_hops() const;
	access_t get_tot_DRAM_acc() const;
	energy_t get_tot_DRAM_cost() const;
	access_t get_DRAM_acc(mlen_t i) const;
	energy_t get_hop_cost() const;
	energy_t get_NoC_hop_cost() const;
	energy_t get_NoP_hop_cost() const;
	energy_t get_cost() const;
	cycle_t get_dram_time() const;
	cycle_t get_time() const;
	///cycle_t get_nocp_time() const;
	void unicast(pos_t src, pos_t dst, vol_t size);
	hop_t unicastCalc(pos_t src, pos_t dst, vol_t size);
	// TODO: dst needs to be in inc. order.
	void multicast(pos_t src, const pos_t* dst, cidx_t len, vol_t size);
	hop_t multicastCalc(pos_t src, const pos_t* dst, cidx_t len, vol_t size);
	// DRAM is at (-1,x) and (n,x)
	void unicast_dram(pos_t dst, vol_t size, mlen_t dram_id);
	void unicast_to_dram(pos_t dst, vol_t size, mlen_t dram_id);
	void multicast_dram(const pos_t* dst, cidx_t len, vol_t size, mlen_t dram_id);


	friend std::ostream& operator<<(std::ostream& os, const NoC& noc);

	struct link_info{
		pos_t from, to;
		hop_t total_hops;

		bool operator<(const link_info& other) const;
		bool operator==(const link_info& other) const;
		bool operator>(const link_info& other) const;

		friend std::ostream& operator<<(std::ostream& os, const link_info& info);
	};
	std::vector<link_info> get_link_info() const;
};

#endif // NOC_H
