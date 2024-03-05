#ifndef PLACEMENT_H
#define PLACEMENT_H

#include <cstdint>
#include <memory>
#include <vector>
#include <utility>
#include "datalayout.h"
#include "partition.h"
#include "util.h"
#include "spatial_mapping/light_placement.h"

class Cluster;
//#include "cluster.h"
class Node;
//#include "network.h"

struct PlaceSch{
	PartSch part;
	FetchSch fetch;
	// An order of 1,2,3,0 means B,H,W,K.
	/*struct{
		std::uint8_t o1:2, o2:2, o3:2, o4:2;
	}order;*/
	std::uint8_t order[4];
	std::unique_ptr<DataLayout> ifmLayout, wgtLayout;
	std::unique_ptr<UniqueLayout> ofmLayout/*, memLayout*/;
	std::unique_ptr<pos_t[]> permuteOrder;
	PlaceSch() = default;
	PlaceSch(const PlaceSch& sch);
	PlaceSch(PlaceSch&& sch) = default;
	PlaceSch& operator=(PlaceSch&& sch) = default;
	// PlaceSch(len_t _batch);
	void finalize();
	void initPlacement_by_light(const Cluster& cluster, const Light_placement& place, lid_t layerno);
	void initPlacement(const Cluster& cluster);
	void update(PlaceSch&& sch);
	DataLayout& getIfmL();
	const DataLayout& getIfmL() const;
	DataLayout& getWgtL();
	const DataLayout& getWgtL() const;
	UniqueLayout& getOfmL();
	const UniqueLayout& getOfmL() const;
	friend std::ostream& operator<<(std::ostream& os, const PlaceSch& sch);
};

class PlaceIter;

class PlaceEngine{
public:
	PlaceEngine()=default;
	PlaceIter init(PlaceSch& cur_sch);
}extern placeEngine;

class PlaceIter{
	friend PlaceEngine;
	//const PartSch* cur_part;
	//const Cluster* cur_cluster;
	std::uint8_t perm_len;
	bool hasNext;
	PlaceSch& curSch;
public:
	PlaceIter(PlaceSch& placeSch);
	bool nextPlace(cost_t cost = cost_inf);
	operator bool() const;
};

fmap_range range_from_partition_number(const fmap_shape &shape, len_t batch, const PartSch &partition, cidx_t id);

#endif // PLACEMENT_H
